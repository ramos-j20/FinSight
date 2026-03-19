"""
Airflow DAG: edgar_ingestion

Orchestrates the full SEC EDGAR ingestion pipeline:
  1. Fetch company CIKs for configured tickers
  2. Fetch filing metadata (skip already-ingested filings)
  3. Download raw filing text and store in S3 raw layer
  4. Parse, chunk, and store in S3 processed layer

Airflow Variables required:
  - FINSIGHT_TICKERS:      comma-separated tickers  (e.g. "AAPL,MSFT,NVDA")
  - FINSIGHT_FILING_TYPES: comma-separated types     (e.g. "10-K,10-Q")
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable

log = logging.getLogger(__name__)

# Default DAG arguments
DEFAULT_ARGS = {
    "owner": "finsight",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run_async(coro):
    """Helper to run an async coroutine from synchronous Airflow tasks."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@dag(
    dag_id="edgar_ingestion",
    default_args=DEFAULT_ARGS,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["finsight", "ingestion"],
    doc_md=__doc__,
)
def edgar_ingestion_dag():
    """SEC EDGAR ingestion pipeline."""

    # ------------------------------------------------------------------
    # Task 1: Fetch CIKs
    # ------------------------------------------------------------------
    @task()
    def fetch_company_ciks() -> dict[str, str]:
        """Resolve ticker symbols to SEC CIK numbers."""
        from backend.ingestion.edgar_client import EDGARClient

        tickers_raw = Variable.get("FINSIGHT_TICKERS", default_var="AAPL")
        tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]

        log.info("Fetching CIKs for tickers: %s", tickers)
        client = EDGARClient()

        cik_map: dict[str, str] = {}
        for ticker in tickers:
            try:
                cik = _run_async(client.get_company_cik(ticker))
                cik_map[ticker] = cik
                log.info("CIK resolved: %s -> %s", ticker, cik)
            except Exception:
                log.exception("Failed to fetch CIK for %s", ticker)

        log.info("CIK fetch complete. Resolved %d / %d tickers", len(cik_map), len(tickers))
        return cik_map

    # ------------------------------------------------------------------
    # Task 2: Fetch filing metadata
    # ------------------------------------------------------------------
    @task()
    def fetch_filing_metadata(cik_map: dict[str, str]) -> list[dict]:
        """Fetch filing metadata for each ticker/filing-type combination.

        Skips filings whose raw S3 key already exists (idempotency).
        """
        from backend.ingestion.edgar_client import EDGARClient
        from backend.ingestion.s3_client import S3Client

        filing_types_raw = Variable.get("FINSIGHT_FILING_TYPES", default_var="10-K")
        filing_types = [f.strip() for f in filing_types_raw.split(",") if f.strip()]

        # Convert async URL to sync for DB check
        db_url = settings.DATABASE_URL.replace("+asyncpg", "")
        engine = create_engine(db_url)

        new_filings: list[dict] = []

        with engine.connect() as conn:
            for ticker, cik in cik_map.items():
                for ftype in filing_types:
                    log.info("Fetching %s filings for %s (CIK %s)", ftype, ticker, cik)
                    try:
                        records = _run_async(client.get_filings(cik, ftype, count=5))
                    except Exception:
                        log.exception("Failed to fetch %s filings for %s", ftype, ticker)
                        continue

                    for rec in records:
                        period = rec.period_of_report or rec.filing_date
                        raw_key = S3Client.raw_key(ticker, ftype, period)

                        # Check if record exists in DB
                        res = conn.execute(text("""
                            SELECT 1 FROM public.filing_metadata 
                            WHERE ticker = :t AND filing_type = :f AND period = :p
                        """), {"t": ticker, "f": ftype, "p": period}).fetchone()

                        if res:
                            log.info("Already in DB, skipping: %s %s %s", ticker, ftype, period)
                            continue

                        # If not in DB, we need to ingest (even if already in S3, 
                        # download_and_store_raw will handle it)
                        new_filings.append({
                            "ticker": ticker,
                            "filing_type": ftype,
                            "period": period,
                            "accession_number": rec.accession_number,
                            "filing_date": rec.filing_date,
                            "primary_document_url": rec.primary_document_url,
                            "s3_raw_key": raw_key,
                        })

        log.info("New filings to ingest: %d", len(new_filings))
        return new_filings

    # ------------------------------------------------------------------
    # Task 3: Download raw text and store in S3
    # ------------------------------------------------------------------
    @task()
    def download_and_store_raw(filings: list[dict]) -> list[dict]:
        """Download filing text and upload to the S3 raw layer."""
        from backend.ingestion.edgar_client import EDGARClient, FilingRecord
        from backend.ingestion.s3_client import S3Client

        client = EDGARClient()
        s3 = S3Client()

        stored: list[dict] = []

        for f in filings:
            log.info("Downloading: %s %s %s", f["ticker"], f["filing_type"], f["period"])

            record = FilingRecord(
                accession_number=f["accession_number"],
                filing_date=f["filing_date"],
                period_of_report=f["period"],
                primary_document_url=f["primary_document_url"],
                form_type=f["filing_type"],
            )

            try:
                text = _run_async(client.fetch_filing_text(record))
                s3.upload_text(text, f["s3_raw_key"])
                stored.append(f)
                log.info("Stored raw filing: %s", f["s3_raw_key"])
            except Exception:
                log.exception("Failed %s %s %s", f["ticker"], f["filing_type"], f["period"])

        log.info("Raw filings stored: %d / %d", len(stored), len(filings))
        return stored

    # ------------------------------------------------------------------
    # Task 4: Parse, chunk, and store processed data
    # ------------------------------------------------------------------
    @task()
    def parse_and_chunk(filings: list[dict]) -> None:
        """Parse raw filings, chunk them, and upload to S3 processed layer."""
        from backend.ingestion.chunker import chunk_document, serialize_chunks
        from backend.ingestion.parser import parse_filing_to_text
        from backend.ingestion.s3_client import S3Client
        from sqlalchemy import create_engine, text
        from backend.core.config import get_settings

        s3 = S3Client()
        settings = get_settings()
        db_url = settings.DATABASE_URL
        if "+asyncpg" in db_url:
            db_url = db_url.replace("+asyncpg", "")
        engine = create_engine(db_url)

        for f in filings:
            ticker = f["ticker"]
            ftype = f["filing_type"]
            period = f["period"]
            raw_key = f["s3_raw_key"]

            log.info("Processing: %s %s %s", ticker, ftype, period)

            try:
                raw_text = s3.download_text(raw_key)
                doc = parse_filing_to_text(raw_text, filing_type=ftype, ticker=ticker, period=period)
                chunks = chunk_document(doc)
                processed_key = S3Client.processed_key(ticker, ftype, period)
                s3.upload_text(serialize_chunks(chunks), processed_key)

                # IMPORTANT: Insert tracking record into Postgres for the embedding pipeline
                log.info("Inserting metadata into Postgres for %s", ticker)
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO public.filing_metadata 
                        (ticker, company_name, filing_type, period, s3_raw_key, status, is_embedded, ingested_at, chunk_count)
                        VALUES 
                        (:ticker, 'Company', :ftype, :period, :raw_key, 'COMPLETE', false, NOW(), :c_count)
                    """), {
                        "ticker": ticker, "ftype": ftype, "period": period,
                        "raw_key": raw_key, "c_count": len(chunks)
                    })
                log.info("Successfully inserted metadata for %s", ticker)

                log.info("Done: %s (%d chunks)", processed_key, len(chunks))
            except Exception:
                log.exception("Failed: %s %s %s", ticker, ftype, period)

    # ------------------------------------------------------------------
    # Wire the DAG
    # ------------------------------------------------------------------
    ciks = fetch_company_ciks()
    metadata = fetch_filing_metadata(ciks)
    raw = download_and_store_raw(metadata)
    parse_and_chunk(raw)


# Instantiate the DAG
edgar_ingestion_dag()
