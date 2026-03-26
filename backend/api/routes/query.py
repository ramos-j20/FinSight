"""API routes for query operations."""
import json
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.rag_agent import (
    AgentResponse,
    FinSightAgent,
    QueryRequest as AgentQueryRequest,
)
from backend.api.models import CitationResponse, FeedbackRequest, QueryRequest, QueryResponse
from backend.core.logging import get_logger
from backend.db.models import QueryLog
from backend.db.session import get_db

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_citation_responses(agent_response: AgentResponse) -> list[CitationResponse]:
    """Convert agent Citation dataclasses to API CitationResponse models."""
    return [
        CitationResponse(
            chunk_id=c.chunk_id,
            ticker=c.ticker,
            filing_type=c.filing_type,
            period=c.period,
            excerpt=c.excerpt,
            reference_number=c.reference_number,
        )
        for c in agent_response.citations
    ]


async def _log_query(
    db: AsyncSession,
    agent_req: AgentQueryRequest,
    agent_resp: AgentResponse,
) -> int:
    """Persist a QueryLog row and return its generated id."""
    log = QueryLog(
        session_id=agent_req.session_id,
        query_text=agent_req.query,
        retrieved_chunk_ids=[c.chunk_id for c in agent_resp.retrieved_chunks],
        llm_response=agent_resp.answer,
        citations=[c.dict() for c in agent_resp.citations],
        latency_ms=float(agent_resp.latency_ms),
        model_used=agent_resp.model_used,
        mode_used=agent_resp.mode_used,
        routing_reason=agent_resp.routing_reason,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log.id


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@router.post("/", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Run a RAG query against the FinSight agent and log the result."""
    agent_req = AgentQueryRequest(
        query=request.query,
        session_id=request.session_id,
        ticker_filter=request.ticker_filter,
        filing_type_filter=request.filing_type_filter,
        conversation_history=request.conversation_history,
        mode_override=request.mode_override,
    )

    try:
        agent = FinSightAgent()
        agent_resp = await agent.query(agent_req)
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent query failed", error=str(exc))
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc

    try:
        query_log_id = await _log_query(db, agent_req, agent_resp)
    except Exception as exc:  # noqa: BLE001
        logger.error("Query logging failed", error=str(exc))
        raise HTTPException(status_code=500, detail={"error": "Failed to log query"}) from exc

    return QueryResponse(
        answer=agent_resp.answer,
        citations=_build_citation_responses(agent_resp),
        session_id=agent_resp.session_id,
        latency_ms=agent_resp.latency_ms,
        prompt_version=agent_resp.prompt_version,
        query_log_id=query_log_id,
        model_used=agent_resp.model_used,
        mode_used=agent_resp.mode_used,
        routing_reason=agent_resp.routing_reason,
    )


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------


@router.post("/stream")
async def query_stream_endpoint(request: QueryRequest) -> StreamingResponse:
    """Stream the RAG query response as Server-Sent Events."""

    agent_req = AgentQueryRequest(
        query=request.query,
        session_id=request.session_id,
        ticker_filter=request.ticker_filter,
        filing_type_filter=request.filing_type_filter,
        conversation_history=request.conversation_history,
        mode_override=request.mode_override,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        from backend.db.session import get_session_maker
        session_maker = get_session_maker()
        
        agent = FinSightAgent()
        full_answer = ""
        citations = []
        latency_ms = 0
        model_used = ""
        mode_used = ""
        routing_reason = ""

        try:
            async for chunk in agent.stream_query(agent_req):
                # The final chunk from stream_query is a JSON object with citations.
                if chunk.startswith("\n{") and '"type"' in chunk and '"citations"' in chunk:
                    # Re-format as SSE citations event
                    data_str = chunk.strip()
                    yield f"data: {data_str}\n\n"
                    
                    # Capture metadata for logging
                    try:
                        meta = json.loads(data_str)
                        d = meta.get("data", {})
                        citations = d.get("citations", [])
                        latency_ms = d.get("latency_ms", 0)
                        model_used = d.get("model_used", "")
                        mode_used = d.get("mode_used", "")
                        routing_reason = d.get("routing_reason", "")
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    # Plain text chunk — wrap in SSE text event
                    full_answer += chunk
                    payload = json.dumps({"type": "text", "data": chunk})
                    yield f"data: {payload}\n\n"
            
            # Persist query log after stream finishes using a fresh session
            try:
                async with session_maker() as session:
                    log = QueryLog(
                        session_id=agent_req.session_id,
                        query_text=agent_req.query,
                        retrieved_chunk_ids=[], 
                        llm_response=full_answer,
                        citations=citations,
                        latency_ms=float(latency_ms),
                        model_used=model_used,
                        mode_used=mode_used,
                        routing_reason=routing_reason,
                    )
                    session.add(log)
                    await session.commit()
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to log streaming query to DB", error=str(e))

        except Exception as exc:  # noqa: BLE001
            logger.error("Streaming query failed", error=str(exc))
            error_payload = json.dumps({"type": "error", "data": str(exc)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# POST /query/{query_log_id}/feedback
# ---------------------------------------------------------------------------


@router.post("/{query_log_id}/feedback")
async def submit_feedback(
    query_log_id: int,
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update the feedback score for a logged query."""
    result = await db.execute(select(QueryLog).where(QueryLog.id == query_log_id))
    log = result.scalar_one_or_none()

    if log is None:
        raise HTTPException(status_code=404, detail=f"Query log {query_log_id} not found")

    log.feedback_score = request.score
    await db.commit()

    return {"status": "ok"}
