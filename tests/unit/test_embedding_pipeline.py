"""Unit tests for the embedding pipeline orchestration."""
import os
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from backend.ingestion.chunker import DocumentChunk

MOCK_ENV = {
    "ANTHROPIC_API_KEY": "test",
    "OPENAI_API_KEY": "test",
    "PINECONE_API_KEY": "test",
    "PINECONE_INDEX_NAME": "test",
    "PINECONE_ENVIRONMENT": "test",
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "AWS_S3_BUCKET_NAME": "test",
    "AWS_REGION": "us-east-1",
    "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db",
    "EDGAR_USER_AGENT": "TestAgent test@example.com",
}

SAMPLE_CHUNKS = [
    DocumentChunk(
        chunk_id="AAPL_10-K_2024_0",
        ticker="AAPL",
        filing_type="10-K",
        period="2024",
        text="Apple revenue grew.",
        char_count=19,
        chunk_index=0,
        total_chunks=3,
    ),
    DocumentChunk(
        chunk_id="AAPL_10-K_2024_1",
        ticker="AAPL",
        filing_type="10-K",
        period="2024",
        text="Operating expenses.",
        char_count=18,
        chunk_index=1,
        total_chunks=3,
    ),
    DocumentChunk(
        chunk_id="AAPL_10-K_2024_2",
        ticker="AAPL",
        filing_type="10-K",
        period="2024",
        text="Net income details.",
        char_count=19,
        chunk_index=2,
        total_chunks=3,
    ),
]


def _clear_settings_cache():
    from backend.core.config import get_settings
    get_settings.cache_clear()


@mock.patch.dict(os.environ, MOCK_ENV)
class TestSkipAlreadyEmbedded:
    """Pipeline must skip chunks whose IDs already exist in Pinecone."""

    def setup_method(self):
        _clear_settings_cache()

    @patch("backend.embeddings.pipeline.PineconeClient")
    @patch("backend.embeddings.pipeline.OpenAIEmbedder")
    @patch("backend.embeddings.pipeline.S3Client")
    @patch("backend.embeddings.pipeline.deserialize_chunks")
    def test_skips_existing_chunks(
        self, mock_deser, mock_s3_cls, mock_embedder_cls, mock_pc_cls
    ):
        """If all chunk IDs exist in Pinecone, no embedding or upsert should occur."""
        from backend.embeddings.pipeline import run_embedding_pipeline

        # S3 returns some JSON
        mock_s3 = MagicMock()
        mock_s3_cls.return_value = mock_s3
        mock_s3.download_text.return_value = "[]"

        # Deserialize returns 3 chunks
        mock_deser.return_value = list(SAMPLE_CHUNKS)

        # Pinecone reports all 3 IDs exist
        mock_pc = MagicMock()
        mock_pc_cls.return_value = mock_pc
        mock_pc.fetch_ids.return_value = {
            "AAPL_10-K_2024_0",
            "AAPL_10-K_2024_1",
            "AAPL_10-K_2024_2",
        }

        result = run_embedding_pipeline("AAPL", "10-K", "2024")

        # No embedding should have happened
        mock_embedder_cls.return_value.embed_batch.assert_not_called()
        mock_pc.upsert_chunks.assert_not_called()

        assert result.chunks_embedded == 0
        assert result.vectors_upserted == 0

    @patch("backend.embeddings.pipeline.PineconeClient")
    @patch("backend.embeddings.pipeline.OpenAIEmbedder")
    @patch("backend.embeddings.pipeline.S3Client")
    @patch("backend.embeddings.pipeline.deserialize_chunks")
    def test_embeds_only_new_chunks(
        self, mock_deser, mock_s3_cls, mock_embedder_cls, mock_pc_cls
    ):
        """Only chunks NOT in Pinecone should be embedded and upserted."""
        from backend.embeddings.pipeline import run_embedding_pipeline

        mock_s3 = MagicMock()
        mock_s3_cls.return_value = mock_s3
        mock_s3.download_text.return_value = "[]"

        mock_deser.return_value = list(SAMPLE_CHUNKS)

        # Only chunk 0 exists already
        mock_pc = MagicMock()
        mock_pc_cls.return_value = mock_pc
        mock_pc.fetch_ids.return_value = {"AAPL_10-K_2024_0"}

        # Embedder returns 2 embeddings (for the 2 new chunks)
        mock_embedder = MagicMock()
        mock_embedder_cls.return_value = mock_embedder
        mock_embedder.embed_batch.return_value = [[0.1] * 10, [0.2] * 10]

        # Upsert returns 2
        mock_pc.upsert_chunks.return_value = 2

        result = run_embedding_pipeline("AAPL", "10-K", "2024")

        # Only 2 new chunks should have been embedded
        mock_embedder.embed_batch.assert_called_once()
        texts_sent = mock_embedder.embed_batch.call_args[0][0]
        assert len(texts_sent) == 2

        assert result.chunks_embedded == 2
        assert result.vectors_upserted == 2


@mock.patch.dict(os.environ, MOCK_ENV)
class TestEmbeddingResult:
    """EmbeddingResult must be correctly populated."""

    def setup_method(self):
        _clear_settings_cache()

    @patch("backend.embeddings.pipeline.PineconeClient")
    @patch("backend.embeddings.pipeline.OpenAIEmbedder")
    @patch("backend.embeddings.pipeline.S3Client")
    @patch("backend.embeddings.pipeline.deserialize_chunks")
    def test_result_fields(
        self, mock_deser, mock_s3_cls, mock_embedder_cls, mock_pc_cls
    ):
        """EmbeddingResult should have correct ticker, filing_type, period, and counts."""
        from backend.embeddings.pipeline import EmbeddingResult, run_embedding_pipeline

        mock_s3 = MagicMock()
        mock_s3_cls.return_value = mock_s3
        mock_s3.download_text.return_value = "[]"

        mock_deser.return_value = list(SAMPLE_CHUNKS)

        mock_pc = MagicMock()
        mock_pc_cls.return_value = mock_pc
        mock_pc.fetch_ids.return_value = set()  # nothing exists yet

        mock_embedder = MagicMock()
        mock_embedder_cls.return_value = mock_embedder
        mock_embedder.embed_batch.return_value = [[0.1] * 10] * 3

        mock_pc.upsert_chunks.return_value = 3

        result = run_embedding_pipeline("AAPL", "10-K", "2024")

        assert isinstance(result, EmbeddingResult)
        assert result.ticker == "AAPL"
        assert result.filing_type == "10-K"
        assert result.period == "2024"
        assert result.chunks_embedded == 3
        assert result.vectors_upserted == 3
        assert result.duration_seconds >= 0
        assert isinstance(result.timestamp, datetime)
