"""Evaluation harness for retrieval quality measurement (LLM-free)."""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from backend.core.logging import get_logger

if TYPE_CHECKING:
    from backend.agent.retriever import FinSightRetriever

logger = get_logger(__name__)


@dataclass
class EvalQuery:
    """A single evaluation query with its expected retrieval results."""

    query: str
    ticker_filter: str | None
    expected_chunk_ids: list[str]


@dataclass
class EvalMetrics:
    """Aggregated retrieval evaluation metrics across all queries."""

    hit_rate: float
    mrr: float
    avg_score: float
    total_queries: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def load_eval_dataset(path: str) -> list[EvalQuery]:
    """Load evaluation queries from a JSONL file.

    Each line should be a JSON object with keys:
      - query (str)
      - ticker_filter (str | null)
      - expected_chunk_ids (list[str])

    Args:
        path: Absolute or relative path to the JSONL file.

    Returns:
        List of EvalQuery dataclass instances.
    """
    queries: list[EvalQuery] = []
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                queries.append(
                    EvalQuery(
                        query=data["query"],
                        ticker_filter=data.get("ticker_filter"),
                        expected_chunk_ids=data.get("expected_chunk_ids", []),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping malformed eval dataset line", line_num=line_num, error=str(exc))

    logger.info("Loaded eval dataset", path=path, count=len(queries))
    return queries


def run_retrieval_eval(
    eval_queries: list[EvalQuery],
    retriever: "FinSightRetriever",
    top_k: int = 10,
) -> tuple["EvalMetrics", list[dict]]:
    """Run retrieval evaluation without calling the LLM.

    For each EvalQuery:
    1. Calls retriever.retrieve() with the query and optional ticker filter.
    2. Computes a hit (True if any expected_chunk_id appears in the results).
    3. Finds the rank of the first expected chunk (for MRR).
    4. Accumulates score across retrieved chunks.

    Args:
        eval_queries: List of EvalQuery objects to evaluate.
        retriever: A FinSightRetriever instance.
        top_k: Number of chunks to retrieve per query.

    Returns:
        Tuple of (EvalMetrics, list of per-query result dicts).
    """
    hits = 0
    reciprocal_ranks: list[float] = []
    total_scores: list[float] = []
    per_query_results: list[dict] = []

    for eq in eval_queries:
        retrieved = retriever.retrieve(
            query=eq.query,
            top_k=top_k,
            ticker_filter=eq.ticker_filter,
        )

        retrieved_ids = [c.chunk_id for c in retrieved]
        avg_chunk_score = sum(c.score for c in retrieved) / len(retrieved) if retrieved else 0.0
        total_scores.append(avg_chunk_score)

        hit = any(eid in retrieved_ids for eid in eq.expected_chunk_ids)
        if hit:
            hits += 1

        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in eq.expected_chunk_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        per_query_results.append(
            {
                "query": eq.query,
                "ticker_filter": eq.ticker_filter,
                "expected_chunk_ids": eq.expected_chunk_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "hit": hit,
                "reciprocal_rank": rr,
                "avg_chunk_score": avg_chunk_score,
            }
        )

        logger.debug(
            "Eval query done",
            query=eq.query,
            hit=hit,
            rr=rr,
        )

    n = len(eval_queries)
    metrics = EvalMetrics(
        hit_rate=hits / n if n else 0.0,
        mrr=sum(reciprocal_ranks) / n if n else 0.0,
        avg_score=sum(total_scores) / n if n else 0.0,
        total_queries=n,
    )

    logger.info(
        "Eval complete",
        hit_rate=metrics.hit_rate,
        mrr=metrics.mrr,
        total_queries=n,
    )

    return metrics, per_query_results


async def save_eval_results(
    metrics: EvalMetrics,
    eval_queries: list[EvalQuery],
    results: list[dict],
) -> None:
    """Persist per-query eval results to the EvalResult table.

    Args:
        metrics: Aggregated EvalMetrics for this run.
        eval_queries: The original EvalQuery objects.
        results: Per-query result dicts produced by run_retrieval_eval.
    """
    from sqlalchemy import insert

    from backend.db.models import EvalResult
    from backend.db.session import get_session_maker

    eval_run_id = str(uuid.uuid4())
    session_factory = get_session_maker()

    async with session_factory() as session:
        rows = []
        for result in results:
            rows.append(
                {
                    "eval_run_id": eval_run_id,
                    "query": result["query"],
                    "expected_chunk_ids": result["expected_chunk_ids"],
                    "retrieved_chunk_ids": result["retrieved_chunk_ids"],
                    "hit_rate": float(result["hit"]),
                    "mrr": result["reciprocal_rank"],
                    "faithfulness_score": result["avg_chunk_score"],
                }
            )

        if rows:
            await session.execute(insert(EvalResult), rows)
            await session.commit()

    logger.info(
        "Eval results saved",
        eval_run_id=eval_run_id,
        row_count=len(rows),
    )
