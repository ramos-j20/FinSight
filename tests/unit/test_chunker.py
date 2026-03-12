"""Unit tests for the document chunker."""
import json
import os
from datetime import datetime, timezone
from unittest import mock

from backend.ingestion.chunker import (
    DocumentChunk,
    chunk_document,
    deserialize_chunks,
    serialize_chunks,
)
from backend.ingestion.parser import ParsedDocument

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


def _make_doc(text: str, ticker: str = "AAPL", filing_type: str = "10-K", period: str = "2023-09-30") -> ParsedDocument:
    """Helper to create a ParsedDocument."""
    return ParsedDocument(
        ticker=ticker,
        filing_type=filing_type,
        period=period,
        clean_text=text,
        word_count=len(text.split()),
        char_count=len(text),
        parsed_at=datetime.now(timezone.utc),
    )


def _clear_settings_cache():
    from backend.core.config import get_settings
    get_settings.cache_clear()


@mock.patch.dict(os.environ, MOCK_ENV)
class TestChunkIdDeterminism:
    """chunk_document must produce identical chunk_ids for the same input."""

    def setup_method(self):
        _clear_settings_cache()

    def test_same_input_produces_same_ids(self):
        text = "Apple Inc. " * 500  # ~5500 chars
        doc = _make_doc(text)

        chunks_a = chunk_document(doc, chunk_size=1000, chunk_overlap=200)
        chunks_b = chunk_document(doc, chunk_size=1000, chunk_overlap=200)

        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]

        assert ids_a == ids_b

    def test_chunk_id_format(self):
        text = "Apple Inc. " * 500
        doc = _make_doc(text)

        chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=200)

        for i, chunk in enumerate(chunks):
            expected_id = f"AAPL_10-K_2023-09-30_{i}"
            assert chunk.chunk_id == expected_id


@mock.patch.dict(os.environ, MOCK_ENV)
class TestChunkCount:
    """Chunk count should be reasonable for typical document sizes."""

    def setup_method(self):
        _clear_settings_cache()

    def test_reasonable_count_for_5000_word_doc(self):
        # ~5000 words ≈ 30,000 chars → expect roughly 30-40 chunks at 1000 chars
        text = "Reasonable word. " * 5000
        doc = _make_doc(text)

        chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) > 10
        assert len(chunks) < 200
        # All chunks should report the correct total
        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)


@mock.patch.dict(os.environ, MOCK_ENV)
class TestSerializeDeserialize:
    """Roundtrip serialization must preserve all fields."""

    def setup_method(self):
        _clear_settings_cache()

    def test_roundtrip(self):
        text = "Apple designs and manufactures smartphones. " * 200
        doc = _make_doc(text)

        original_chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=200)
        json_str = serialize_chunks(original_chunks)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

        # Deserialize back
        restored = deserialize_chunks(json_str)

        assert len(restored) == len(original_chunks)
        for orig, rest in zip(original_chunks, restored):
            assert orig.chunk_id == rest.chunk_id
            assert orig.text == rest.text
            assert orig.ticker == rest.ticker
            assert orig.filing_type == rest.filing_type
            assert orig.period == rest.period
            assert orig.char_count == rest.char_count
            assert orig.chunk_index == rest.chunk_index
            assert orig.total_chunks == rest.total_chunks
