"""Embedding pipeline orchestration — end-to-end S3 → embed → Pinecone."""
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.core.logging import get_logger
from backend.embeddings.embedder import OpenAIEmbedder
from backend.embeddings.pinecone_client import PineconeClient
from backend.ingestion.chunker import deserialize_chunks
from backend.ingestion.s3_client import S3Client

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Summary of an embedding pipeline run."""

    ticker: str
    filing_type: str
    period: str
    chunks_embedded: int
    vectors_upserted: int
    duration_seconds: float
    timestamp: datetime


def run_embedding_pipeline(
    ticker: str,
    filing_type: str,
    period: str,
) -> EmbeddingResult:
    """Run the full embedding pipeline for a single filing.

    Flow:
        1. Download processed chunks JSON from S3.
        2. Deserialize to ``list[DocumentChunk]``.
        3. Filter out chunk IDs already in Pinecone (idempotency).
        4. Batch-embed remaining chunks via OpenAI.
        5. Upsert vectors to Pinecone.
        6. Return an ``EmbeddingResult`` summary.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        filing_type: SEC filing type (e.g. "10-K").
        period: Filing period string (e.g. "2024").

    Returns:
        An EmbeddingResult dataclass with run statistics.
    """
    start = time.time()
    logger.info(
        "Starting embedding pipeline",
        ticker=ticker,
        filing_type=filing_type,
        period=period,
    )

    # 1. Read chunks from S3
    s3 = S3Client()
    s3_key = S3Client.processed_key(ticker, filing_type, period)
    json_str = s3.download_text(s3_key)

    # 2. Deserialize
    all_chunks = deserialize_chunks(json_str)
    logger.info("Deserialized chunks", count=len(all_chunks))

    # 3. Deduplicate — skip chunks already in Pinecone
    pc = PineconeClient()
    all_ids = [c.chunk_id for c in all_chunks]
    existing_ids = pc.fetch_ids(all_ids)

    new_chunks = [c for c in all_chunks if c.chunk_id not in existing_ids]
    skipped = len(all_chunks) - len(new_chunks)
    if skipped:
        logger.info("Skipped already-embedded chunks", skipped=skipped)

    if not new_chunks:
        logger.info("All chunks already embedded, nothing to do")
        return EmbeddingResult(
            ticker=ticker,
            filing_type=filing_type,
            period=period,
            chunks_embedded=0,
            vectors_upserted=0,
            duration_seconds=round(time.time() - start, 2),
            timestamp=datetime.now(timezone.utc),
        )

    # 4. Embed
    embedder = OpenAIEmbedder()
    texts = [c.text for c in new_chunks]
    embeddings = embedder.embed_batch(texts)
    logger.info("Batch embedding complete", count=len(embeddings))

    # 5. Upsert
    upserted = pc.upsert_chunks(new_chunks, embeddings)

    duration = round(time.time() - start, 2)
    result = EmbeddingResult(
        ticker=ticker,
        filing_type=filing_type,
        period=period,
        chunks_embedded=len(new_chunks),
        vectors_upserted=upserted,
        duration_seconds=duration,
        timestamp=datetime.now(timezone.utc),
    )

    logger.info(
        "Embedding pipeline complete",
        ticker=ticker,
        chunks_embedded=result.chunks_embedded,
        vectors_upserted=result.vectors_upserted,
        duration_seconds=result.duration_seconds,
    )
    return result
