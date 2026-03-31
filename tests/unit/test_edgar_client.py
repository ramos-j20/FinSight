"""Unit tests for the SEC EDGAR API client."""
import os
from unittest import mock

import httpx
import pytest

from backend.ingestion.edgar_client import EDGARClient, FilingRecord


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

SAMPLE_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

SAMPLE_HTML_FILING = """
<html>
<head><title>10-K Filing</title></head>
<body>
<div>
<p>ITEM 1. BUSINESS</p>
<p>Apple Inc. designs, manufactures, and markets smartphones.</p>
<p>ITEM 1A. RISK FACTORS</p>
<p>The Company is subject to various risks.</p>
</div>
</body>
</html>
"""


def _clear_settings_cache():
    """Clear the lru_cache on get_settings so mock env vars take effect."""
    from backend.core.config import get_settings
    get_settings.cache_clear()


@mock.patch.dict(os.environ, MOCK_ENV)
class TestGetCompanyCik:
    """Tests for EDGARClient.get_company_cik."""

    def setup_method(self):
        _clear_settings_cache()

    @pytest.mark.asyncio
    async def test_returns_cik_for_valid_ticker(self):
        """get_company_cik returns a zero-padded CIK for a known ticker."""
        mock_response = httpx.Response(
            status_code=200,
            json=SAMPLE_TICKERS_JSON,
            request=httpx.Request("GET", "https://example.com"),
        )
        client = EDGARClient()
        with mock.patch.object(client, "_get", return_value=mock_response):
            cik = await client.get_company_cik("AAPL")

        assert cik == "0000320193"

    @pytest.mark.asyncio
    async def test_raises_for_unknown_ticker(self):
        """get_company_cik raises EDGARFetchError for an unknown ticker."""
        from backend.core.exceptions import EDGARFetchError

        mock_response = httpx.Response(
            status_code=200,
            json=SAMPLE_TICKERS_JSON,
            request=httpx.Request("GET", "https://example.com"),
        )
        client = EDGARClient()
        with mock.patch.object(client, "_get", return_value=mock_response):
            with pytest.raises(EDGARFetchError, match="not found"):
                await client.get_company_cik("ZZZZ")


@mock.patch.dict(os.environ, MOCK_ENV)
class TestFetchFilingText:
    """Tests for EDGARClient.fetch_filing_text."""

    def setup_method(self):
        _clear_settings_cache()

    @pytest.mark.asyncio
    async def test_strips_html_tags(self):
        """fetch_filing_text should strip HTML and return plain text."""
        mock_response = httpx.Response(
            status_code=200,
            text=SAMPLE_HTML_FILING,
            request=httpx.Request("GET", "https://example.com"),
        )
        record = FilingRecord(
            accession_number="0000320193-24-000001",
            filing_date="2024-01-01",
            period_of_report="2023-09-30",
            primary_document_url="https://example.com/filing.htm",
            form_type="10-K",
        )
        client = EDGARClient()
        with mock.patch.object(client, "_get", return_value=mock_response):
            text = await client.fetch_filing_text(record)

        assert "<html>" not in text
        assert "<body>" not in text
        assert "<p>" not in text
        assert "<div>" not in text

        assert "ITEM 1. BUSINESS" in text
        assert "Apple Inc." in text
        assert "RISK FACTORS" in text

    @pytest.mark.asyncio
    async def test_handles_plain_text(self):
        """fetch_filing_text should pass through plain text unchanged."""
        plain = "ITEM 1. BUSINESS\n\nApple Inc. designs smartphones."
        mock_response = httpx.Response(
            status_code=200,
            text=plain,
            request=httpx.Request("GET", "https://example.com"),
        )
        record = FilingRecord(
            accession_number="0000320193-24-000001",
            filing_date="2024-01-01",
            period_of_report="2023-09-30",
            primary_document_url="https://example.com/filing.txt",
            form_type="10-K",
        )
        client = EDGARClient()
        with mock.patch.object(client, "_get", return_value=mock_response):
            text = await client.fetch_filing_text(record)

        assert "ITEM 1. BUSINESS" in text
        assert "Apple Inc." in text


@mock.patch.dict(os.environ, MOCK_ENV)
class TestRetryLogic:
    """Tests for the retry logic on 429 responses."""

    def setup_method(self):
        _clear_settings_cache()

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        """_get should retry on HTTP 429 and succeed on subsequent attempt."""
        request = httpx.Request("GET", "https://example.com")
        err_response = httpx.Response(status_code=429, request=request)
        ok_response = httpx.Response(
            status_code=200,
            json=SAMPLE_TICKERS_JSON,
            request=request,
        )

        call_count = 0

        async def mock_get(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError(
                    "Rate limited",
                    request=request,
                    response=err_response,
                )
            return ok_response

        client = EDGARClient()

        with mock.patch("httpx.AsyncClient.get", side_effect=mock_get):
            response = await client._get("https://example.com")

        assert response.status_code == 200
        assert call_count == 2  # first call 429, second call 200
