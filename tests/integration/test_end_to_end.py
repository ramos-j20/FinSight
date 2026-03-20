"""End-to-end integration test for the FinSight API.

Run as a standalone script:
    python tests/integration/test_end_to_end.py

Requires a running backend (docker-compose up, or uvicorn locally).
"""
import os
import sys
import time

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
QUERY = "What are Apple's main business risks?"
TICKER = "AAPL"


def _check(condition: bool, message: str) -> None:
    if not condition:
        print(f"  ❌ FAIL: {message}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✅ {message}")


def main() -> None:
    print("=" * 60)
    print("FinSight – End-to-End Integration Test")
    print(f"Backend: {BACKEND_URL}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Health check
    # ------------------------------------------------------------------
    print("\n[1/5] Health check …")
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=10)
        _check(r.status_code == 200, f"GET /health → {r.status_code}")
    except Exception as exc:
        print(f"  ❌ Backend unreachable: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. POST /query — runs a real RAG query
    # ------------------------------------------------------------------
    print(f'\n[2/5] POST /query — "{QUERY}" …')
    t0 = time.monotonic()
    try:
        r = httpx.post(
            f"{BACKEND_URL}/query/",
            json={
                "query": QUERY,
                "ticker_filter": TICKER,
            },
            timeout=120,
        )
    except Exception as exc:
        print(f"  ❌ Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    _check(r.status_code == 200, f"POST /query → {r.status_code}")
    body = r.json()
    _check("answer" in body, "Response has 'answer' field")
    _check(bool(body.get("answer")), "Answer is non-empty")
    _check("citations" in body, "Response has 'citations' field")
    _check("query_log_id" in body, "Response has 'query_log_id' field")
    _check(isinstance(body.get("citations"), list), "citations is a list")

    query_log_id: int = body["query_log_id"]
    citations: list = body["citations"]
    latency_ms: int = body.get("latency_ms", elapsed_ms)

    # ------------------------------------------------------------------
    # 3. POST /query/{id}/feedback
    # ------------------------------------------------------------------
    print(f"\n[3/5] POST /query/{query_log_id}/feedback (score=5) …")
    r = httpx.post(
        f"{BACKEND_URL}/query/{query_log_id}/feedback",
        json={"query_log_id": query_log_id, "score": 5},
        timeout=10,
    )
    _check(r.status_code == 200, f"POST /query/{query_log_id}/feedback → {r.status_code}")
    _check(r.json().get("status") == "ok", "Feedback status is 'ok'")

    # ------------------------------------------------------------------
    # 4. GET /eval/results
    # ------------------------------------------------------------------
    print("\n[4/5] GET /eval/results …")
    r = httpx.get(f"{BACKEND_URL}/eval/results", timeout=10)
    _check(r.status_code == 200, f"GET /eval/results → {r.status_code}")
    results = r.json()
    _check(isinstance(results, list), "eval/results returns a list")

    # ------------------------------------------------------------------
    # 5. GET /documents — confirm index has at least some content
    # ------------------------------------------------------------------
    print("\n[5/5] GET /documents …")
    r = httpx.get(f"{BACKEND_URL}/documents/", timeout=10)
    _check(r.status_code == 200, f"GET /documents → {r.status_code}")
    docs = r.json()
    _check("total_count" in docs, "documents response has 'total_count'")
    _check("filings" in docs, "documents response has 'filings'")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(
        f"✅ End-to-end test passed. "
        f"Latency: {latency_ms}ms. "
        f"Citations: {len(citations)}."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
