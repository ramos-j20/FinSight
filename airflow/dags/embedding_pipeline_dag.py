"""
Airflow DAG: embedding_pipeline

Orchestrates the embedding pipeline for SEC filings:
  1. Query PostgreSQL for filings with status=COMPLETE that haven't been embedded yet.
  2. For each pending filing, run the embedding pipeline (S3 → OpenAI → Pinecone).
  3. Log Pinecone index stats.

Runs weekly after edgar_ingestion completes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "finsight",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="embedding_pipeline",
    default_args=DEFAULT_ARGS,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["finsight", "embeddings"],
    doc_md=__doc__,
)
def embedding_pipeline_dag():
    """Embedding pipeline — embed SEC filings and upsert to Pinecone."""

    @task()
    def get_pending_filings() -> list[dict]:
        """Query FilingMetadata for COMPLETE filings not yet embedded.

        Uses synchronous psycopg2 (Airflow tasks are synchronous).
        """
        from sqlalchemy import create_engine, text

        from backend.core.config import get_settings

        settings = get_settings()
        # Convert async URL to sync for Airflow tasks
        db_url = settings.DATABASE_URL
        if "+asyncpg" in db_url:
            db_url = db_url.replace("+asyncpg", "")

        engine = create_engine(db_url)

        with engine.connect() as conn:
            stmt = text("""
                SELECT id, ticker, filing_type, period 
                FROM public.filing_metadata 
                WHERE status = 'COMPLETE' AND is_embedded = false
            """)
            rows = conn.execute(stmt).fetchall()

            filings = [
                {
                    "id": row[0],
                    "ticker": row[1],
                    "filing_type": row[2],
                    "period": row[3],
                }
                for row in rows
            ]

        log.info("Pending filings for embedding: %d", len(filings))
        return filings

    @task()
    def embed_and_upsert(filing: dict) -> dict | None:
        """Run the embedding pipeline for a single pending filing.

        Marks is_embedded=True on success.
        """
        from sqlalchemy import create_engine, text

        from backend.core.config import get_settings
        from backend.embeddings.pipeline import run_embedding_pipeline

        settings = get_settings()
        db_url = settings.DATABASE_URL
        if "+asyncpg" in db_url:
            db_url = db_url.replace("+asyncpg", "")

        engine = create_engine(db_url)

        ticker = filing["ticker"]
        filing_type = filing["filing_type"]
        period = filing["period"]
        filing_id = filing["id"]

        log.info("Embedding: %s %s %s", ticker, filing_type, period)
        try:
            result = run_embedding_pipeline(ticker, filing_type, period)

            # Mark as embedded in PostgreSQL
            with engine.begin() as conn:
                stmt = text("""
                    UPDATE filing_metadata 
                    SET is_embedded = true 
                    WHERE id = :id
                """)
                conn.execute(stmt, {"id": filing_id})

            log.info(
                "Embedded %s %s %s: %d chunks, %d vectors, %.1fs",
                ticker, filing_type, period,
                result.chunks_embedded, result.vectors_upserted,
                result.duration_seconds,
            )
            return {
                "ticker": ticker,
                "filing_type": filing_type,
                "period": period,
                "chunks_embedded": result.chunks_embedded,
                "vectors_upserted": result.vectors_upserted,
                "duration_seconds": result.duration_seconds,
            }
        except Exception:
            log.exception("Failed to embed %s %s %s", ticker, filing_type, period)
            return None

    @task()
    def log_index_stats(results: list[dict | None]) -> None:
        """Log current Pinecone index statistics."""
        from backend.embeddings.pinecone_client import PineconeClient

        # Filter out None values from failed tasks
        valid_results = [r for r in results if r is not None]

        pc = PineconeClient()
        stats = pc.index_stats()
        log.info("Pinecone index stats: %s", stats)
        log.info("Embedding results summary: %s", valid_results)

    pending = get_pending_filings()
    embedded = embed_and_upsert.expand(filing=pending)
    log_index_stats(embedded)


# Instantiate the DAG
embedding_pipeline_dag()
