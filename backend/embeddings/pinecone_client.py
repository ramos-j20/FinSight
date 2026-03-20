"""Pinecone vector-database client for FinSight."""
from dataclasses import dataclass

from pinecone import Pinecone

from backend.core.config import get_settings
from backend.core.exceptions import PineconeUpsertError
from backend.core.logging import get_logger
from backend.ingestion.chunker import DocumentChunk

logger = get_logger(__name__)

# Pinecone recommends upsert batches of ≤ 100 vectors.
_UPSERT_BATCH_SIZE = 100


@dataclass
class RetrievedChunk:
    """A chunk retrieved from a Pinecone similarity query."""

    chunk_id: str
    ticker: str
    filing_type: str
    period: str
    text: str
    score: float
    source_url: str


class PineconeClient:
    """Wrapper around the Pinecone SDK for vector operations.

    Connects to the index specified by ``PINECONE_INDEX_NAME`` on
    instantiation.
    """

    def __init__(self) -> None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = pc.Index(settings.PINECONE_INDEX_NAME)
        logger.info(
            "Pinecone client initialised",
            index=settings.PINECONE_INDEX_NAME,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert document-chunk vectors into Pinecone.

        Vectors are batched in groups of 100.  Each vector carries
        metadata fields required for filtered retrieval.

        Args:
            chunks: The document chunks to store.
            embeddings: Corresponding embedding vectors (same order).

        Returns:
            The number of vectors successfully upserted.

        Raises:
            PineconeUpsertError: If the upsert fails.
        """
        if len(chunks) != len(embeddings):
            raise PineconeUpsertError(
                f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
            )

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "ticker": chunk.ticker,
                    "filing_type": chunk.filing_type,
                    "period": chunk.period,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_url": getattr(chunk, "source_url", ""),
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                },
            })

        upserted = 0
        try:
            for i in range(0, len(vectors), _UPSERT_BATCH_SIZE):
                batch = vectors[i : i + _UPSERT_BATCH_SIZE]
                self._index.upsert(vectors=batch)
                upserted += len(batch)
                logger.info(
                    "Upserted batch",
                    batch_start=i,
                    batch_size=len(batch),
                    total_upserted=upserted,
                )
        except Exception as exc:
            raise PineconeUpsertError(
                f"Failed to upsert vectors (upserted {upserted} before error): {exc}"
            ) from exc

        logger.info("Upsert complete", total_upserted=upserted)
        return upserted

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Query Pinecone for the most similar chunks.

        Args:
            embedding: The query embedding vector.
            top_k: Number of results to return.
            filter: Optional Pinecone metadata filter.

        Returns:
            A list of RetrievedChunk objects ordered by descending score.
        """
        query_params: dict = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filter is not None:
            query_params["filter"] = filter

        results = self._index.query(**query_params)

        retrieved: list[RetrievedChunk] = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            retrieved.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", match["id"]),
                    ticker=meta.get("ticker", ""),
                    filing_type=meta.get("filing_type", ""),
                    period=meta.get("period", ""),
                    text=meta.get("text", ""),
                    score=match.get("score", 0.0),
                    source_url=meta.get("source_url", ""),
                )
            )

        return retrieved

    def fetch_ids(self, ids: list[str]) -> set[str]:
        """Check which vector IDs already exist in the index.

        Uses the Pinecone ``fetch`` endpoint to retrieve vectors by ID
        and returns the set of IDs that were found.

        Args:
            ids: List of vector IDs to check.

        Returns:
            Set of IDs that already exist in the index.
        """
        if not ids:
            return set()

        existing: set[str] = set()
        # Pinecone fetch supports up to 1000 IDs per call, but AWS ELB throws 414 
        # Request-URI Too Large if the query string is too long. Since our IDs are long,
        # we reduce the batch size to 200.
        for i in range(0, len(ids), 200):
            batch = ids[i : i + 200]
            response = self._index.fetch(ids=batch)
            if hasattr(response, "vectors") and response.vectors:
                existing.update(response.vectors.keys())
            elif isinstance(response, dict) and "vectors" in response:
                existing.update(response["vectors"].keys())

        return existing

    def index_stats(self) -> dict:
        """Return Pinecone index statistics.

        Returns:
            Dictionary with vector count, dimension, and namespace info.
        """
        stats = self._index.describe_index_stats()
        logger.info("Pinecone index stats", stats=str(stats))
        return stats
