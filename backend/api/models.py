"""Pydantic v2 API models for FinSight request/response schemas."""
import uuid
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming query from an API client."""

    query: str = Field(..., min_length=3, max_length=500)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker_filter: str | None = None
    filing_type_filter: Literal["10-K", "10-Q"] | None = None
    conversation_history: list[dict] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """User feedback on a specific query response."""

    query_log_id: int
    score: int = Field(..., ge=1, le=5)
    comment: str | None = None


class EvalRunRequest(BaseModel):
    """Request to trigger an evaluation run."""

    eval_dataset_path: str


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CitationResponse(BaseModel):
    """A single citation returned inside a QueryResponse."""

    chunk_id: str
    ticker: str
    filing_type: str
    period: str
    excerpt: str
    reference_number: int


class QueryResponse(BaseModel):
    """Full response to a /query request."""

    answer: str
    citations: list[CitationResponse]
    session_id: str
    latency_ms: int
    prompt_version: str
    query_log_id: int


class StreamChunkResponse(BaseModel):
    """A single chunk in an SSE stream."""

    type: Literal["text", "citations"]
    data: str | list


class FilingMetadataResponse(BaseModel):
    """Metadata for a single SEC filing."""

    id: int
    ticker: str
    filing_type: str
    period: str
    status: str
    chunk_count: int
    is_embedded: bool
    ingested_at: str  # ISO-formatted datetime string

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of filings."""

    filings: list[FilingMetadataResponse]
    total_count: int
