"""Unit tests for PineconeClient."""
import os
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


def _clear_settings_cache():
    from backend.core.config import get_settings
    get_settings.cache_clear()


def _make_chunks(count: int) -> list[DocumentChunk]:
    """Generate a list of dummy DocumentChunk objects."""
    return [
        DocumentChunk(
            chunk_id=f"AAPL_10-K_2024_{i}",
            ticker="AAPL",
            filing_type="10-K",
            period="2024",
            text=f"Chunk text number {i}",
            char_count=20,
            chunk_index=i,
            total_chunks=count,
        )
        for i in range(count)
    ]


@mock.patch.dict(os.environ, MOCK_ENV)
class TestUpsertBatching:
    """upsert_chunks must batch vectors in groups of ≤ 100."""

    def setup_method(self):
        _clear_settings_cache()

    @patch("backend.embeddings.pinecone_client.Pinecone")
    def test_single_batch_under_100(self, mock_pinecone_cls):
        """50 chunks should result in a single upsert call."""
        from backend.embeddings.pinecone_client import PineconeClient

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        chunks = _make_chunks(50)
        embeddings = [[0.1] * 10 for _ in range(50)]

        client = PineconeClient()
        result = client.upsert_chunks(chunks, embeddings)

        assert result == 50
        assert mock_index.upsert.call_count == 1

        call_args = mock_index.upsert.call_args
        batch = call_args.kwargs.get("vectors") or call_args[1].get("vectors")
        assert len(batch) == 50

    @patch("backend.embeddings.pinecone_client.Pinecone")
    def test_multiple_batches_over_100(self, mock_pinecone_cls):
        """250 chunks should result in 3 upsert calls (100 + 100 + 50)."""
        from backend.embeddings.pinecone_client import PineconeClient

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        chunks = _make_chunks(250)
        embeddings = [[0.1] * 10 for _ in range(250)]

        client = PineconeClient()
        result = client.upsert_chunks(chunks, embeddings)

        assert result == 250
        assert mock_index.upsert.call_count == 3

        batch_sizes = []
        for call in mock_index.upsert.call_args_list:
            batch = call.kwargs.get("vectors") or call[1].get("vectors")
            batch_sizes.append(len(batch))

        assert batch_sizes == [100, 100, 50]

    @patch("backend.embeddings.pinecone_client.Pinecone")
    def test_exactly_100(self, mock_pinecone_cls):
        """Exactly 100 chunks should result in a single upsert call."""
        from backend.embeddings.pinecone_client import PineconeClient

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        chunks = _make_chunks(100)
        embeddings = [[0.1] * 10 for _ in range(100)]

        client = PineconeClient()
        result = client.upsert_chunks(chunks, embeddings)

        assert result == 100
        assert mock_index.upsert.call_count == 1


@mock.patch.dict(os.environ, MOCK_ENV)
class TestQuery:
    """query must return correctly structured RetrievedChunk objects."""

    def setup_method(self):
        _clear_settings_cache()

    @patch("backend.embeddings.pinecone_client.Pinecone")
    def test_query_returns_retrieved_chunks(self, mock_pinecone_cls):
        """Query should map Pinecone results to RetrievedChunk dataclass."""
        from backend.embeddings.pinecone_client import PineconeClient, RetrievedChunk

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "AAPL_10-K_2024_0",
                    "score": 0.95,
                    "metadata": {
                        "chunk_id": "AAPL_10-K_2024_0",
                        "ticker": "AAPL",
                        "filing_type": "10-K",
                        "period": "2024",
                        "text": "Apple revenue grew significantly.",
                        "source_url": "https://sec.gov/example",
                    },
                },
                {
                    "id": "AAPL_10-K_2024_1",
                    "score": 0.87,
                    "metadata": {
                        "chunk_id": "AAPL_10-K_2024_1",
                        "ticker": "AAPL",
                        "filing_type": "10-K",
                        "period": "2024",
                        "text": "Operating expenses increased.",
                        "source_url": "",
                    },
                },
            ],
        }

        client = PineconeClient()
        results = client.query(embedding=[0.1] * 10, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

        assert results[0].chunk_id == "AAPL_10-K_2024_0"
        assert results[0].ticker == "AAPL"
        assert results[0].filing_type == "10-K"
        assert results[0].period == "2024"
        assert results[0].score == 0.95
        assert results[0].text == "Apple revenue grew significantly."
        assert results[0].source_url == "https://sec.gov/example"

        assert results[1].chunk_id == "AAPL_10-K_2024_1"
        assert results[1].score == 0.87
