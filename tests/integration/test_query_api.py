"""
Integration tests for the FinSight query API.

These tests hit the LIVE backend at http://localhost:8000 (or the BASE_URL
env var). Start the stack first:

    docker compose up --build -d

Then run these tests from INSIDE the backend container:

    docker compose exec backend python -m pytest tests/integration/test_query_api.py -v

Or from your host machine (requires httpx installed locally):

    python -m pytest tests/integration/test_query_api.py -v
"""
import os

import httpx
import pytest

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def client():
    """Synchronous httpx client pointing at the live API."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_check_returns_ok(self, client):
        """GET /health should return 200 and status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_post_query_returns_200_with_correct_shape(self, client):
        """POST /query should return 200 with the full QueryResponse schema."""
        resp = client.post(
            "/query/",
            json={"query": "What are Apple's main revenue segments?"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert isinstance(data["citations"], list)
        assert "session_id" in data
        assert "latency_ms" in data
        assert "prompt_version" in data
        assert "query_log_id" in data
        assert isinstance(data["query_log_id"], int)

    def test_post_query_with_ticker_filter(self, client):
        """POST /query with a ticker_filter should call the agent with that filter."""
        resp = client.post(
            "/query/",
            json={
                "query": "What are Apple's main revenue segments?",
                "ticker_filter": "AAPL",
                "filing_type_filter": "10-K",
            },
        )
        assert resp.status_code == 200, resp.text

    def test_post_query_too_short_returns_422(self, client):
        """POST /query with a query shorter than 3 chars should return 422."""
        resp = client.post("/query/", json={"query": "AB"})
        assert resp.status_code == 422

    def test_post_query_missing_query_field_returns_422(self, client):
        """POST /query with no query field should return 422."""
        resp = client.post("/query/", json={})
        assert resp.status_code == 422

    def test_post_query_too_long_returns_422(self, client):
        """POST /query with query > 500 chars should return 422."""
        resp = client.post("/query/", json={"query": "x" * 501})
        assert resp.status_code == 422

    def test_feedback_endpoint_updates_record(self, client):
        """POST /query/{id}/feedback should return status ok."""
        # First create a real log entry
        resp = client.post(
            "/query/",
            json={"query": "What is Apple's revenue breakdown?"},
        )
        assert resp.status_code == 200, resp.text
        query_log_id = resp.json()["query_log_id"]

        # Now submit feedback
        feedback_resp = client.post(
            f"/query/{query_log_id}/feedback",
            json={"query_log_id": query_log_id, "score": 5},
        )
        assert feedback_resp.status_code == 200
        assert feedback_resp.json()["status"] == "ok"

    def test_feedback_on_nonexistent_log_returns_404(self, client):
        """POST /query/{id}/feedback for a non-existent log should return 404."""
        resp = client.post(
            "/query/999999999/feedback",
            json={"query_log_id": 999999999, "score": 3},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------


class TestQueryStreamEndpoint:
    def test_stream_returns_event_stream_content_type(self, client):
        """POST /query/stream should return Content-Type: text/event-stream."""
        with client.stream(
            "POST",
            "/query/stream",
            json={"query": "What is Apple's operating income?"},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            # Read at least a few bytes to verify something streams
            chunks = []
            for line in resp.iter_lines():
                if line:
                    chunks.append(line)
                if len(chunks) >= 3:
                    break
            assert len(chunks) > 0, "Expected at least one SSE chunk"


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


class TestDocumentsEndpoint:
    def test_get_documents_returns_200_with_pagination(self, client):
        """GET /documents should return 200 with filings list and total_count."""
        resp = client.get("/documents/")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "filings" in data
        assert "total_count" in data
        assert isinstance(data["filings"], list)
        assert isinstance(data["total_count"], int)

    def test_get_documents_with_ticker_filter(self, client):
        """GET /documents?ticker=AAPL should return only AAPL filings."""
        resp = client.get("/documents/", params={"ticker": "AAPL"})
        assert resp.status_code == 200
        data = resp.json()
        for filing in data["filings"]:
            assert filing["ticker"] == "AAPL"

    def test_get_documents_pagination(self, client):
        """GET /documents with page_size=1 should return at most 1 filing."""
        resp = client.get("/documents/", params={"page": 1, "page_size": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["filings"]) <= 1

    def test_get_documents_page_size_too_large_returns_422(self, client):
        """GET /documents with page_size > 100 should return 422."""
        resp = client.get("/documents/", params={"page_size": 200})
        assert resp.status_code == 422

    def test_get_ticker_filings_returns_list(self, client):
        """GET /documents/{ticker}/filings should return a list."""
        resp = client.get("/documents/AAPL/filings")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
