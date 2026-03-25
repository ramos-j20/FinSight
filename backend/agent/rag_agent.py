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


class FinSightAgent:
    """Core RAG agent orchestrating retrieval, prompting, and formatting."""
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = FinSightRetriever()
        self.client = Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.async_client = AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.model_used = self.settings.ANTHROPIC_MODEL
        
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

    def query(self, query_request: QueryRequest) -> AgentResponse:
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
        system_prompt = build_system_prompt()
        
        if self._is_comparative(query_request.query):
            ticker, periods = self._extract_ticker_and_periods(query_request.query, query_request.ticker_filter)
            messages = build_comparison_prompt(
                query=query_request.query,
                context=context,
                ticker=ticker,
                periods=periods
            )
        else:
            messages = build_rag_prompt(
                query=query_request.query,
                context=context,
                conversation_history=query_request.conversation_history
            )
            
        response = self.client.messages.create(
            model=route.model,
            max_tokens=route.max_tokens,
            system=system_prompt,
            messages=messages
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
            routing_reason=route.routing_reason
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
        system_prompt = build_system_prompt()
        
        if self._is_comparative(query_request.query):
            ticker, periods = self._extract_ticker_and_periods(query_request.query, query_request.ticker_filter)
            messages = build_comparison_prompt(
                query=query_request.query,
                context=context,
                ticker=ticker,
                periods=periods
            )
        else:
            messages = build_rag_prompt(
                query=query_request.query,
                context=context,
                conversation_history=query_request.conversation_history
            )
            
        stream = await self.async_client.messages.create(
            model=route.model,
            max_tokens=route.max_tokens,
            system=system_prompt,
            messages=messages,
            stream=True
        )
        
        full_text = ""
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                chunk_text = event.delta.text
                full_text += chunk_text
                yield chunk_text
                
        try:
            citations = extract_citations(full_text, chunks)
        except CitationParseError as exc:
            logger.error("Citation parsing failed during streaming", error=str(exc))
            citations = []
            
        latency_ms = int((time.time() - start_time) * 1000)
        final_chunk = {
            "type": "citations",
            "data": {
                "citations": [citation.dict() for citation in citations],
                "latency_ms": latency_ms,
                "prompt_version": PROMPT_VERSION,
                "model_used": route.model,
                "mode_used": route.mode_used,
                "routing_reason": route.routing_reason
            }
        }
        
        # Optionally, format the final json clearly
        yield "\n" + json.dumps(final_chunk)
