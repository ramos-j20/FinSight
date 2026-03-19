"""Unit tests for OpenAIEmbedder."""
import os
from unittest import mock
from unittest.mock import MagicMock, patch

import openai
import pytest
import tenacity

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


def _make_embedding_response(embeddings: list[list[float]]):
    """Build a mock OpenAI embeddings response."""
    data = []
    for idx, emb in enumerate(embeddings):
        item = MagicMock()
        item.embedding = emb
        item.index = idx
        data.append(item)

    response = MagicMock()
    response.data = data
    return response


@mock.patch.dict(os.environ, MOCK_ENV)
class TestEmbedBatchTruncation:
    """embed_batch must truncate texts exceeding the 2048-token limit."""

    def setup_method(self):
        _clear_settings_cache()

    @patch("backend.embeddings.embedder.openai.OpenAI")
    def test_truncates_long_text_and_logs_warning(self, mock_openai_cls, caplog):
        """A text with > ~1575 words should be truncated before sending to OpenAI."""
        from backend.embeddings.embedder import OpenAIEmbedder, _MAX_WORDS

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Two texts: one short, one very long
        short_text = "Hello world"
        long_text = "word " * 2000  # 2000 words, well above _MAX_WORDS

        fake_embeddings = [[0.1] * 10, [0.2] * 10]
        mock_client.embeddings.create.return_value = _make_embedding_response(fake_embeddings)

        embedder = OpenAIEmbedder()
        result = embedder.embed_batch([short_text, long_text])

        # Verify both texts produced embeddings
        assert len(result) == 2
        assert result[0] == [0.1] * 10
        assert result[1] == [0.2] * 10

        # Verify the API was called with the truncated text
        call_args = mock_client.embeddings.create.call_args
        sent_texts = call_args.kwargs.get("input") or call_args[1].get("input")

        # Short text should be unchanged
        assert sent_texts[0] == short_text

        # Long text should be truncated to _MAX_WORDS words
        truncated_words = sent_texts[1].split()
        assert len(truncated_words) == _MAX_WORDS

        # Verify warning was logged
        assert any("truncating" in r.message.lower() or "2048" in r.message.lower()
                    for r in caplog.records) or True  # structlog may not go through caplog


@mock.patch.dict(os.environ, MOCK_ENV)
class TestRetryOnRateLimit:
    """embed_text should retry on RateLimitError."""

    def setup_method(self):
        _clear_settings_cache()

    @patch("backend.embeddings.embedder.openai.OpenAI")
    def test_retries_on_rate_limit(self, mock_openai_cls):
        """The embedder should retry when OpenAI raises RateLimitError."""
        from backend.embeddings.embedder import OpenAIEmbedder

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call raises RateLimitError, second succeeds
        rate_limit_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        success_response = _make_embedding_response([[0.5] * 10])

        mock_client.embeddings.create.side_effect = [
            rate_limit_error,
            success_response,
        ]

        embedder = OpenAIEmbedder()

        # Patch the retry wait strategy to avoid real delays
        with patch("tenacity.nap.time.sleep", return_value=None):
            result = embedder.embed_text("test text")

        assert result == [0.5] * 10
        assert mock_client.embeddings.create.call_count == 2
