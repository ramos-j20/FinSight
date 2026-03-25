"""API routes for evaluation operations."""
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from backend.api.models import EvalRunRequest
from backend.core.logging import get_logger
from backend.db.session import get_session_maker
from backend.eval.harness import EvalMetrics, load_eval_dataset, run_retrieval_eval, save_eval_results

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /eval/run
# ---------------------------------------------------------------------------


@router.post("/run", response_model=None)
async def run_eval(request: EvalRunRequest) -> dict:
    """Run the retrieval evaluation harness on a JSONL dataset.

    Args:
        request: Contains the path to the JSONL eval dataset.

    Returns:
        EvalMetrics dict with hit_rate, mrr, avg_score, total_queries, timestamp.
    """
    from backend.agent.retriever import FinSightRetriever

    try:
        eval_queries = load_eval_dataset(request.eval_dataset_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Eval dataset not found: {request.eval_dataset_path}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to load eval dataset: {exc}") from exc

    if not eval_queries:
        raise HTTPException(status_code=400, detail="Eval dataset is empty")

    try:
        retriever = FinSightRetriever()
        metrics, results = run_retrieval_eval(eval_queries, retriever)
    except Exception as exc:  # noqa: BLE001
        logger.error("Eval run failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Eval run failed: {exc}") from exc

    try:
        await save_eval_results(metrics, eval_queries, results)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save eval results to DB", error=str(exc))
        # Non-fatal: still return the metrics even if persistence fails

    return {
        "hit_rate": metrics.hit_rate,
        "mrr": metrics.mrr,
        "avg_score": metrics.avg_score,
        "total_queries": metrics.total_queries,
        "timestamp": metrics.timestamp,
    }


# ---------------------------------------------------------------------------
# GET /eval/results
# ---------------------------------------------------------------------------


@router.get("/results")
async def get_eval_results() -> list[dict]:
    """Return the last 20 evaluation run results from the database."""
    from backend.db.models import EvalResult

    session_factory = get_session_maker()
    async with session_factory() as session:
        stmt = (
            select(EvalResult)
            .order_by(EvalResult.created_at.desc())
            .limit(20)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "id": row.id,
            "eval_run_id": row.eval_run_id,
            "query": row.query,
            "expected_chunk_ids": row.expected_chunk_ids,
            "retrieved_chunk_ids": row.retrieved_chunk_ids,
            "hit_rate": row.hit_rate,
            "mrr": row.mrr,
            "faithfulness_score": row.faithfulness_score,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# GET /eval/routing-stats
# ---------------------------------------------------------------------------


@router.get("/routing-stats")
async def get_routing_stats() -> list[dict]:
    """Return model routing statistics from query logs."""
    from backend.db.models import QueryLog
    from sqlalchemy import func

    session_factory = get_session_maker()
    async with session_factory() as session:
        stmt = (
            select(
                QueryLog.mode_used,
                QueryLog.model_used,
                func.count(QueryLog.id).label("query_count"),
                func.avg(QueryLog.latency_ms).label("avg_latency")
            )
            .group_by(QueryLog.mode_used, QueryLog.model_used)
        )
        result = await session.execute(stmt)
        rows = result.all()

    return [
        {
            "mode_used": mode or "unknown",
            "model_used": model or "unknown",
            "count": q_count,
            "avg_latency": float(avg_lat) if avg_lat else 0.0,
        }
        for mode, model, q_count, avg_lat in rows
    ]
