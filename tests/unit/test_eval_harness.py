"""Unit tests for the retrieval evaluation harness."""
import pytest
from unittest.mock import MagicMock

from backend.eval.harness import EvalMetrics, EvalQuery, run_retrieval_eval
from backend.embeddings.pinecone_client import RetrievedChunk


def _make_chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        ticker="AAPL",
        filing_type="10-K",
        period="2023",
        text=f"Text for chunk {chunk_id}",
        score=score,
        source_url="",
    )


def _make_retriever(returned_chunks: list[RetrievedChunk]) -> MagicMock:
    """Return a mock retriever that returns the given chunks on .retrieve()."""
    retriever = MagicMock()
    retriever.retrieve.return_value = returned_chunks
    return retriever


class TestHitRate:
    def test_hit_rate_one_when_all_expected_in_results(self):
        """hit_rate should be 1.0 when all expected chunks are retrieved."""
        chunks = [
            _make_chunk("chunk-a"),
            _make_chunk("chunk-b"),
            _make_chunk("chunk-c"),
        ]
        retriever = _make_retriever(chunks)
        queries = [
            EvalQuery(query="Revenue?", ticker_filter="AAPL", expected_chunk_ids=["chunk-a"]),
            EvalQuery(query="Risks?", ticker_filter="AAPL", expected_chunk_ids=["chunk-b"]),
        ]

        metrics, _ = run_retrieval_eval(queries, retriever, top_k=10)

        assert metrics.hit_rate == 1.0

    def test_hit_rate_zero_when_no_expected_in_results(self):
        """hit_rate should be 0.0 when retrieved chunks don't match expected."""
        chunks = [_make_chunk("chunk-x"), _make_chunk("chunk-y")]
        retriever = _make_retriever(chunks)
        queries = [
            EvalQuery(query="Revenue?", ticker_filter="AAPL", expected_chunk_ids=["chunk-different-1"]),
            EvalQuery(query="Risks?", ticker_filter="AAPL", expected_chunk_ids=["chunk-different-2"]),
        ]

        metrics, _ = run_retrieval_eval(queries, retriever, top_k=10)

        assert metrics.hit_rate == 0.0


class TestMRR:
    def test_mrr_one_when_expected_chunk_is_rank_1(self):
        """MRR should be 1.0 when the expected chunk is the first retrieved result."""
        chunks = [
            _make_chunk("chunk-target"),  # rank 1
            _make_chunk("chunk-other-a"),
            _make_chunk("chunk-other-b"),
        ]
        retriever = _make_retriever(chunks)
        queries = [
            EvalQuery(query="Revenue?", ticker_filter="AAPL", expected_chunk_ids=["chunk-target"]),
        ]

        metrics, _ = run_retrieval_eval(queries, retriever, top_k=10)

        assert metrics.mrr == 1.0

    def test_mrr_point_five_when_expected_chunk_is_rank_2(self):
        """MRR should be 0.5 when the expected chunk is the second retrieved result."""
        chunks = [
            _make_chunk("chunk-other"),  # rank 1
            _make_chunk("chunk-target"),  # rank 2 → RR = 0.5
            _make_chunk("chunk-other-b"),
        ]
        retriever = _make_retriever(chunks)
        queries = [
            EvalQuery(query="Revenue?", ticker_filter="AAPL", expected_chunk_ids=["chunk-target"]),
        ]

        metrics, _ = run_retrieval_eval(queries, retriever, top_k=10)

        assert metrics.mrr == pytest.approx(0.5)


class TestEvalMetricsShape:
    def test_metrics_totals_match_query_count(self):
        """total_queries in EvalMetrics should equal the number of eval queries passed."""
        chunks = [_make_chunk("chunk-z")]
        retriever = _make_retriever(chunks)
        queries = [
            EvalQuery(query="Q1", ticker_filter=None, expected_chunk_ids=["chunk-z"]),
            EvalQuery(query="Q2", ticker_filter=None, expected_chunk_ids=["chunk-z"]),
            EvalQuery(query="Q3", ticker_filter=None, expected_chunk_ids=["chunk-z"]),
        ]

        metrics, results = run_retrieval_eval(queries, retriever, top_k=10)

        assert metrics.total_queries == 3
        assert len(results) == 3
