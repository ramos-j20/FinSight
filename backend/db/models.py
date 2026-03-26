"""SQLAlchemy database models for FinSight."""
import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, JSON, DateTime, Float, Integer, String, Enum as SQLAlchemyEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""
    pass


class IngestionStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class FilingMetadata(Base):
    __tablename__ = "filing_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    company_name: Mapped[str] = mapped_column(String(255))
    filing_type: Mapped[str] = mapped_column(String(50))  # e.g., 10-K, 10-Q
    period: Mapped[str] = mapped_column(String(50))
    s3_raw_key: Mapped[str] = mapped_column(String(1024))
    s3_processed_key: Mapped[str] = mapped_column(String(1024), nullable=True)
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.utcnow
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[IngestionStatus] = mapped_column(
        SQLAlchemyEnum(IngestionStatus), default=IngestionStatus.PENDING
    )
    is_embedded: Mapped[bool] = mapped_column(Boolean, default=False)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), index=True)
    query_text: Mapped[str] = mapped_column(String)
    retrieved_chunk_ids: Mapped[list[Any]] = mapped_column(JSON)
    llm_response: Mapped[str] = mapped_column(String)
    citations: Mapped[list[Any]] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.utcnow
    )
    feedback_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    mode_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    routing_reason: Mapped[str | None] = mapped_column(String, nullable=True)


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    eval_run_id: Mapped[str] = mapped_column(String(255), index=True)
    query: Mapped[str] = mapped_column(String)
    expected_chunk_ids: Mapped[list[Any]] = mapped_column(JSON)
    retrieved_chunk_ids: Mapped[list[Any]] = mapped_column(JSON)
    hit_rate: Mapped[float] = mapped_column(Float)
    mrr: Mapped[float] = mapped_column(Float)
    faithfulness_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.utcnow
    )


class InferenceMetrics(Base):
    __tablename__ = "inference_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_log_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_used: Mapped[str] = mapped_column(String(100))
    mode_used: Mapped[str] = mapped_column(String(50))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer)
    estimated_cost_usd: Mapped[float] = mapped_column(Float)  # Using Float for simplicity in models, but precision matters in logic
    caching_enabled: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.utcnow
    )


class BatchJobMetrics(Base):
    __tablename__ = "batch_job_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_job_id: Mapped[str] = mapped_column(String(255), index=True)
    total_requests: Mapped[int] = mapped_column(Integer)
    succeeded: Mapped[int] = mapped_column(Integer)
    failed: Mapped[int] = mapped_column(Integer)
    input_tokens_total: Mapped[int] = mapped_column(Integer)
    output_tokens_total: Mapped[int] = mapped_column(Integer)
    estimated_cost_usd: Mapped[float] = mapped_column(Float)
    estimated_cost_without_batch_usd: Mapped[float] = mapped_column(Float)
    savings_usd: Mapped[float] = mapped_column(Float)
    savings_pct: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50))  # PENDING / PROCESSING / COMPLETE / FAILED
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.utcnow
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
