"""Core RAG agent for FinSight."""
import json
import time
import re
from dataclasses import dataclass, field
from typing import AsyncGenerator

from anthropic import Anthropic, AsyncAnthropic

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.embeddings.pinecone_client import RetrievedChunk
from backend.agent.retriever import FinSightRetriever
from backend.agent.prompt_builder import (
    build_system_prompt, 
    build_rag_prompt, 
    build_comparison_prompt,
    PROMPT_VERSION
)
from backend.agent.citation_parser import extract_citations, Citation, CitationParseError
from backend.agent.model_router import resolve_model
from backend.core.cost_calculator import calculate_cost, format_cost_report
from backend.db.models import InferenceMetrics
from backend.db.session import get_session_maker
from backend.db.session import get_session_maker


logger = get_logger(__name__)


@dataclass
class QueryRequest:
    """Request object for the FinSightAgent query."""
    query: str
    session_id: str
    ticker_filter: str | None = None
    filing_type_filter: str | None = None
    conversation_history: list[dict] = field(default_factory=list)
    mode_override: str | None = None
    enable_caching: bool = True


@dataclass
class AgentResponse:
    """Response object containing the answer, citations, and metadata."""
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    session_id: str
    query: str
    latency_ms: int
    prompt_version: str
    model_used: str
    mode_used: str
    routing_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cache_hit: bool = False


class FinSightAgent:
    """Core RAG agent orchestrating retrieval, prompting, and formatting."""
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = FinSightRetriever()
        self.client = Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.async_client = AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.model_used = self.settings.ANTHROPIC_MODEL
        self.session_maker = get_session_maker()

    async def _persist_metrics(
        self,
        model_used: str,
        mode_used: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        latency_ms: int,
        estimated_cost_usd: float,
        caching_enabled: bool,
        query_log_id: int | None = None
    ) -> None:
        """Persist inference metrics to the database."""
        try:
            async with self.session_maker() as session:
                metrics = InferenceMetrics(
                    query_log_id=query_log_id,
                    model_used=model_used,
                    mode_used=mode_used,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    latency_ms=latency_ms,
                    estimated_cost_usd=estimated_cost_usd,
                    caching_enabled=caching_enabled
                )
                session.add(metrics)
                await session.commit()
        except Exception as e:
            logger.error("Failed to persist inference metrics", error=str(e))
        
    def _is_comparative(self, query: str) -> bool:
        """Detect if the query is comparative."""
        keywords = ["vs", "compared to", "year over year", "quarter over quarter"]
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)
        
    def _extract_ticker_and_periods(self, query: str, ticker_filter: str | None) -> tuple[str, list[str]]:
        """Extract the ticker and a list of periods for the comparison prompt if needed."""
        ticker = ticker_filter if ticker_filter else "the requested company"
        
        periods = []
        q_matches = re.finditer(r'Q\d\s\d{4}', query, re.IGNORECASE)
        for m in q_matches:
            periods.append(m.group(0))
            
        y_matches = re.finditer(r'\b20\d{2}\b', query)
        for m in y_matches:
            if m.group(0) not in [p.split()[-1] for p in periods if ' ' in p]:
                periods.append(m.group(0))
                
        if not periods:
            periods = ["Period 1", "Period 2"]
            
        return ticker, periods

    async def query(self, query_request: QueryRequest) -> AgentResponse:
        """Run a standard RAG query using the Claude API."""
        start_time = time.time()
        
        query_type = "comparison" if self._is_comparative(query_request.query) else "rag"
        route = resolve_model(
            mode_override=query_request.mode_override,
            query=query_request.query,
            query_type=query_type
        )
        
        logger.info(
            "Starting query", 
            session_id=query_request.session_id, 
            query=query_request.query,
            model_selected=route.model,
            mode=route.mode_used,
            reason=route.routing_reason
        )
        
        target_top_k = max(route.top_k, getattr(self.settings, "RETRIEVAL_TOP_K", 10))
        chunks = self.retriever.retrieve(
            query=query_request.query,
            top_k=target_top_k,
            ticker_filter=query_request.ticker_filter,
            filing_type_filter=query_request.filing_type_filter
        )
        
        context = self.retriever.format_context(chunks)
        system_prompt = build_system_prompt(enable_caching=True)
        
        if self._is_comparative(query_request.query):
            ticker, periods = self._extract_ticker_and_periods(query_request.query, query_request.ticker_filter)
            messages = build_comparison_prompt(
                query=query_request.query,
                context=context,
                ticker=ticker,
                periods=periods,
                enable_caching=True
            )
        else:
            messages = build_rag_prompt(
                query=query_request.query,
                context=context,
                conversation_history=query_request.conversation_history,
                enable_caching=True
            )
            
        response = await self.async_client.messages.create(
            model=route.model,
            max_tokens=route.max_tokens,
            system=system_prompt,
            messages=messages
        )
        
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0)
        
        cost_breakdown = calculate_cost(
            model=route.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens
        )
        estimated_cost_usd = float(cost_breakdown.total_cost)
        cache_hit = cache_read_tokens > 0

        latency_ms = int((time.time() - start_time) * 1000)

        # Log metrics at INFO level as requested
        logger.info(
            f"[METRICS] model={route.model} | tokens_in={input_tokens} | tokens_out={output_tokens} | "
            f"cache_read={cache_read_tokens} | cache_write={cache_write_tokens} | "
            f"cost=${estimated_cost_usd:.6f} | cache_hit={cache_hit}"
        )

        await self._persist_metrics(
            model_used=route.model,
            mode_used=route.mode_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            caching_enabled=query_request.enable_caching
        )
        
        answer_text = response.content[0].text
        
        try:
            citations = extract_citations(answer_text, chunks)
        except CitationParseError as exc:
            logger.error("Citation parsing failed", error=str(exc))
            citations = []
            
        latency_ms = int((time.time() - start_time) * 1000)
        
        agent_resp = AgentResponse(
            answer=answer_text,
            citations=citations,
            retrieved_chunks=chunks,
            session_id=query_request.session_id,
            query=query_request.query,
            latency_ms=latency_ms,
            prompt_version=PROMPT_VERSION,
            model_used=route.model,
            mode_used=route.mode_used,
            routing_reason=route.routing_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            estimated_cost_usd=estimated_cost_usd,
            cache_hit=cache_hit
        )
        
        logger.info(
            "Query complete",
            session_id=query_request.session_id,
            latency_ms=latency_ms,
            citation_count=len(citations)
        )
        
        return agent_resp

    async def stream_query(self, query_request: QueryRequest) -> AsyncGenerator[str, None]:
        """Stream a query, yielding text chunks and finally a JSON with citations."""
        start_time = time.time()
        
        query_type = "comparison" if self._is_comparative(query_request.query) else "rag"
        route = resolve_model(
            mode_override=query_request.mode_override,
            query=query_request.query,
            query_type=query_type
        )
        
        logger.info(
            "Starting stream_query", 
            session_id=query_request.session_id, 
            query=query_request.query,
            model_selected=route.model,
            mode=route.mode_used,
            reason=route.routing_reason
        )
        
        target_top_k = max(route.top_k, getattr(self.settings, "RETRIEVAL_TOP_K", 10))
        chunks = self.retriever.retrieve(
            query=query_request.query,
            top_k=target_top_k,
            ticker_filter=query_request.ticker_filter,
            filing_type_filter=query_request.filing_type_filter
        )
        
        context = self.retriever.format_context(chunks)
        system_prompt = build_system_prompt(enable_caching=True)
        
        if self._is_comparative(query_request.query):
            ticker, periods = self._extract_ticker_and_periods(query_request.query, query_request.ticker_filter)
            messages = build_comparison_prompt(
                query=query_request.query,
                context=context,
                ticker=ticker,
                periods=periods,
                enable_caching=True
            )
        else:
            messages = build_rag_prompt(
                query=query_request.query,
                context=context,
                conversation_history=query_request.conversation_history,
                enable_caching=True
            )
            
        stream = await self.async_client.messages.create(
            model=route.model,
            max_tokens=route.max_tokens,
            system=system_prompt,
            messages=messages,
            stream=True
        )
        
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        full_text = ""
        
        async for event in stream:
            if event.type == "message_start":
                usage = event.message.usage
                input_tokens = usage.input_tokens
                cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)
                cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0)
            elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                chunk_text = event.delta.text
                full_text += chunk_text
                yield chunk_text
            elif event.type == "message_delta":
                output_tokens = event.usage.output_tokens

        cost_breakdown = calculate_cost(
            model=route.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens
        )
        estimated_cost_usd = float(cost_breakdown.total_cost)
        cache_hit = cache_read_tokens > 0

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[METRICS] model={route.model} | tokens_in={input_tokens} | tokens_out={output_tokens} | "
            f"cache_read={cache_read_tokens} | cache_write={cache_write_tokens} | "
            f"cost=${estimated_cost_usd:.6f} | cache_hit={cache_hit}"
        )

        await self._persist_metrics(
            model_used=route.model,
            mode_used=route.mode_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            caching_enabled=True
        )
                
        try:
            citations = extract_citations(full_text, chunks)
        except CitationParseError as exc:
            logger.error("Citation parsing failed during streaming", error=str(exc))
            citations = []
            
        final_chunk = {
            "type": "citations",
            "data": {
                "citations": [citation.dict() for citation in citations],
                "latency_ms": latency_ms,
                "prompt_version": PROMPT_VERSION,
                "model_used": route.model,
                "mode_used": route.mode_used,
                "routing_reason": route.routing_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "cache_hit": cache_hit
            }
        }
        
        yield "\n" + json.dumps(final_chunk)
