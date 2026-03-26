"""Batch API processor for FinSight using Anthropic's Message Batch API."""
import time
import asyncio
import datetime
import json
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.cost_calculator import calculate_cost, CostBreakdown, format_cost_report
from backend.agent.retriever import FinSightRetriever
from backend.agent.prompt_builder import build_system_prompt, build_rag_prompt
from backend.db.models import BatchJobMetrics
from backend.db.session import get_session_maker

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    custom_id: str
    query: str
    ticker_filter: str | None = None
    filing_type_filter: str | None = None


@dataclass
class BatchResult:
    custom_id: str
    success: bool
    answer: str | None = None
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_breakdown: CostBreakdown | None = None


class FinSightBatchProcessor:
    """Processor for handling bulk RAG queries via Anthropic's Batch API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.async_client = AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.retriever = FinSightRetriever()
        self.session_maker = get_session_maker()
        self.model = "claude-haiku-4-5"  # As per task requirement

    async def submit_batch(self, requests: list[BatchRequest], job_label: str) -> str:
        """Submit a batch of requests to Anthropic."""
        batch_requests = []
        
        for req in requests:
            # 1. Retrieve context
            chunks = self.retriever.retrieve(
                query=req.query,
                ticker_filter=req.ticker_filter,
                filing_type_filter=req.filing_type_filter,
                top_k=getattr(self.settings, "RETRIEVAL_TOP_K", 10)
            )
            context = self.retriever.format_context(chunks)
            
            # 2. Build prompt (caching disabled for batch)
            system_prompt = build_system_prompt(enable_caching=False)
            messages = build_rag_prompt(
                query=req.query,
                context=context,
                conversation_history=[],
                enable_caching=False
            )
            
            # 3. Format for Anthropic Batch API
            batch_requests.append({
                "custom_id": req.custom_id,
                "params": {
                    "model": self.model,
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": messages
                }
            })

        # Submit to Anthropic
        batch = await self.async_client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        
        # Persist BatchJobMetrics record
        try:
            async with self.session_maker() as session:
                metrics = BatchJobMetrics(
                    batch_job_id=batch_id,
                    total_requests=len(requests),
                    succeeded=0,
                    failed=0,
                    input_tokens_total=0,
                    output_tokens_total=0,
                    estimated_cost_usd=0.0,
                    estimated_cost_without_batch_usd=0.0,
                    savings_usd=0.0,
                    savings_pct=0.0,
                    duration_seconds=0,
                    status="PENDING",
                    created_at=datetime.datetime.utcnow()
                )
                session.add(metrics)
                await session.commit()
        except Exception as e:
            logger.error("Failed to persist BatchJobMetrics", error=str(e))

        logger.info(f"[BATCH] Submitted {len(requests)} requests. batch_job_id={batch_id}")
        return batch_id

    async def poll_batch(self, batch_job_id: str) -> str:
        """Poll the status of a batch job and update DB."""
        batch = await self.async_client.messages.batches.retrieve(batch_job_id)
        status = batch.processing_status
        
        try:
            async with self.session_maker() as session:
                from sqlalchemy import select
                res = await session.execute(select(BatchJobMetrics).where(BatchJobMetrics.batch_job_id == batch_job_id))
                metrics = res.scalar_one_or_none()
                if metrics:
                    metrics.status = status.upper()
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update BatchJobMetrics status for {batch_job_id}", error=str(e))

        logger.info(f"[BATCH] Status={status} for batch_job_id={batch_job_id}")
        return status

    async def retrieve_results(self, batch_job_id: str) -> list[BatchResult]:
        """Stream results, aggregate metrics, and update DB."""
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        succeeded = 0
        failed = 0
        
        # Start timer for duration calculation
        start_time = time.time()
        
        # Note: results() is a streaming generator, so we use async for
        async for result in await self.async_client.messages.batches.results(batch_job_id):
            custom_id = result.custom_id
            
            if result.result.type == "succeeded":
                succeeded += 1
                message = result.result.message
                answer = message.content[0].text
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                cost_breakdown = calculate_cost(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    is_batch=True
                )
                
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=True,
                    answer=answer,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_breakdown=cost_breakdown
                ))
            else:
                failed += 1
                error_msg = str(result.result.error) if hasattr(result.result, "error") else "Unknown error"
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=False,
                    error_message=error_msg
                ))

        # Aggregate totals
        agg_breakdown = calculate_cost(
            model=self.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            is_batch=True
        )
        
        duration = int(time.time() - start_time)
        
        try:
            async with self.session_maker() as session:
                from sqlalchemy import select
                res = await session.execute(select(BatchJobMetrics).where(BatchJobMetrics.batch_job_id == batch_job_id))
                metrics = res.scalar_one_or_none()
                if metrics:
                    metrics.succeeded = succeeded
                    metrics.failed = failed
                    metrics.input_tokens_total = total_input_tokens
                    metrics.output_tokens_total = total_output_tokens
                    metrics.estimated_cost_usd = float(agg_breakdown.total_cost)
                    metrics.estimated_cost_without_batch_usd = float(agg_breakdown.total_cost_without_optimizations)
                    metrics.savings_usd = float(agg_breakdown.total_savings)
                    metrics.savings_pct = float(agg_breakdown.savings_pct)
                    metrics.duration_seconds = duration
                    metrics.status = "COMPLETE"
                    metrics.completed_at = datetime.datetime.utcnow()
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to finalize BatchJobMetrics for {batch_job_id}", error=str(e))

        logger.info(format_cost_report(agg_breakdown))
        return results

    async def run_batch_eval(self, requests: list[BatchRequest]) -> Any:
        """Main entry point for batch processing with polling."""
        batch_id = await self.submit_batch(requests, "eval_job")
        
        max_wait = 24 * 3600  # 24 hours
        waited = 0
        poll_interval = 30
        
        while waited < max_wait:
            status = await self.poll_batch(batch_id)
            if status in ["ended", "completed", "failed", "canceled"]:
                break
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            
        await self.retrieve_results(batch_id)
        
        # Return final metrics record
        async with self.session_maker() as session:
            from sqlalchemy import select
            res = await session.execute(select(BatchJobMetrics).where(BatchJobMetrics.batch_job_id == batch_id))
            return res.scalar_one_or_none()
