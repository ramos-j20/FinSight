"""Benchmark script for FinSight cost optimization features."""
import argparse
import asyncio
import datetime
import os
import sys
import time
from decimal import Decimal

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agent.rag_agent import FinSightAgent, QueryRequest
from backend.agent.batch_processor import FinSightBatchProcessor, BatchRequest
from backend.db.session import get_session_maker
from backend.db.models import InferenceMetrics, BatchJobMetrics
from sqlalchemy import select, func


def print_header():
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║          FINSIGHT OPTIMIZATION BENCHMARK REPORT          ║")
    print("  ╚══════════════════════════════════════════════════════════╝")


async def run_benchmark(num_queries: int, ticker: str, filing_type: str):
    agent = FinSightAgent()
    batch_processor = FinSightBatchProcessor()
    session_maker = get_session_maker()
    
    model = "claude-haiku-4-5"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    report_filename = f"reports/benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    queries = [
        f"What are the main risk factors for {ticker} in their {filing_type}?",
        f"How much did {ticker}'s revenue change in the latest {filing_type}?",
        f"What is the net income for {ticker} reported in the {filing_type}?",
        f"Analyze the liquidity of {ticker} based on the {filing_type}.",
        f"What are the significant legal proceedings mentioned for {ticker}?"
    ] * (num_queries // 5 + 1)
    queries = queries[:num_queries]

    # --- PHASE 1: BASELINE ---
    print(f"\nRunning PHASE 1: BASELINE ({num_queries} queries)...")
    baseline_metrics = []
    for i, q in enumerate(queries):
        req = QueryRequest(
            query=q, 
            session_id=f"bench_baseline_{i}", 
            ticker_filter=ticker, 
            filing_type_filter=filing_type,
            enable_caching=False,
            mode_override="fast" # Force Haiku for benchmark to stay under Sonnet rate limits
        )
        resp = await agent.query(req)
        baseline_metrics.append(resp)
        await asyncio.sleep(5) # Stay within 30k TPM limit

    # --- PHASE 2: CACHING ---
    print(f"Running PHASE 2: CACHING ({num_queries} queries)...")
    caching_metrics = []
    # Warm the cache first
    q_warm = queries[0]
    await agent.query(QueryRequest(query=q_warm, session_id="bench_warm", ticker_filter=ticker, filing_type_filter=filing_type, mode_override="fast"))
    await asyncio.sleep(5)
    
    for i, q in enumerate(queries):
        req = QueryRequest(
            query=q, 
            session_id=f"bench_caching_{i}", 
            ticker_filter=ticker, 
            filing_type_filter=filing_type,
            mode_override="fast"
        )
        resp = await agent.query(req)
        caching_metrics.append(resp)
        await asyncio.sleep(5) # Stay within 30k TPM limit

    # --- PHASE 3: BATCH API ---
    print(f"Running PHASE 3: BATCH API ({num_queries} requests)...")
    batch_reqs = [
        BatchRequest(custom_id=f"batch_{i}", query=q, ticker_filter=ticker, filing_type_filter=filing_type)
        for i, q in enumerate(queries)
    ]
    batch_job_metrics = await batch_processor.run_batch_eval(batch_reqs)

    # --- AGGREGATE RESULTS ---
    # Phase 1
    p1_avg_cost = sum(m.estimated_cost_usd for m in baseline_metrics) / num_queries
    p1_avg_latency = sum(m.latency_ms for m in baseline_metrics) / num_queries
    p1_total_cost = sum(m.estimated_cost_usd for m in baseline_metrics)
    
    # Phase 2
    p2_avg_cost = sum(m.estimated_cost_usd for m in caching_metrics) / num_queries
    p2_avg_latency = sum(m.latency_ms for m in caching_metrics) / num_queries
    p2_total_cost = sum(m.estimated_cost_usd for m in caching_metrics)
    p2_hits = sum(1 for m in caching_metrics if m.cache_hit)
    p2_hit_rate = (p2_hits / num_queries) * 100
    
    cost_red = ((p1_avg_cost - p2_avg_cost) / p1_avg_cost * 100) if p1_avg_cost > 0 else 0
    lat_red = ((p1_avg_latency - p2_avg_latency) / p1_avg_latency * 100) if p1_avg_latency > 0 else 0
    saved_vs_baseline = p1_total_cost - p2_total_cost

    # Phase 3
    p3_total_cost = batch_job_metrics.estimated_cost_usd if batch_job_metrics else 0
    p3_cost_no_batch = batch_job_metrics.estimated_cost_without_batch_usd if batch_job_metrics else 0
    p3_saved = batch_job_metrics.savings_usd if batch_job_metrics else 0
    p3_saved_pct = batch_job_metrics.savings_pct if batch_job_metrics else 0
    p3_duration = batch_job_metrics.duration_seconds if batch_job_metrics else 0

    # --- FORMAT REPORT ---
    report = f"""
  Queries run: {num_queries} | Ticker: {ticker} | Filing type: {filing_type} | Model: {model}
  Timestamp: {timestamp}

  ── PHASE 1: BASELINE (no optimizations) ──────────────────
  Avg cost per query:   ${p1_avg_cost:.6f}
  Avg latency:          {p1_avg_latency:,.0f}ms
  Total cost ({num_queries}q):     ${p1_total_cost:.6f}
  Cache hit rate:       0%

  ── PHASE 2: PROMPT CACHING ───────────────────────────────
  Avg cost per query:   ${p2_avg_cost:.6f}
  Avg latency:          {p2_avg_latency:,.0f}ms
  Total cost ({num_queries}q):     ${p2_total_cost:.6f}
  Cache hit rate:       {p2_hit_rate:.0f}%
  Cost reduction:       {cost_red:+.1f}%
  Latency reduction:    {lat_red:+.1f}%
  Saved vs baseline:    ${saved_vs_baseline:.6f}

  ── PHASE 3: BATCH API ────────────────────────────────────
  Total requests:       {num_queries}
  Succeeded:            {batch_job_metrics.succeeded if batch_job_metrics else 0}
  Total cost (batch):   ${p3_total_cost:.6f}
  Cost without batch:   ${p3_cost_no_batch:.6f}
  Saved via batch:      ${p3_saved:.6f} ({p3_saved_pct:.1f}%)
  Duration:             {p3_duration}s

  ── COMBINED SAVINGS SUMMARY ──────────────────────────────
  Caching alone saves:       {cost_red:.1f}% per query
  Batch alone saves:         {p3_saved_pct:.1f}% per query
  Best strategy by use case:
    Real-time queries  → Prompt Caching  (low latency, ~{cost_red:.0f}% cheaper)
    Bulk / async jobs  → Batch API       ({p3_saved_pct:.0f}% cheaper, async)

All results have been persisted to the inference_metrics and batch_job_metrics tables.
"""
    
    print_header()
    print(report)
    
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("╔══════════════════════════════════════════════════════════╗\n")
        f.write("║          FINSIGHT OPTIMIZATION BENCHMARK REPORT          ║\n")
        f.write("╚══════════════════════════════════════════════════════════╝\n")
        f.write(report)
    
    print(f"Report saved to {report_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FinSight optimization benchmarks.")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries to run per phase")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker for benchmark")
    parser.add_argument("--filing-type", type=str, default="10-K", help="Filing type for benchmark")
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.queries, args.ticker, args.filing_type))
