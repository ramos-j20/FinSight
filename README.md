# FinSight — Earnings Intelligence Platform

FinSight is a production-grade Retrieval-Augmented Generation (RAG) system built on top of SEC EDGAR filings. It ingests 10-K and 10-Q documents, embeds them into a Pinecone vector index, and exposes a FastAPI backend and Streamlit chat interface so analysts can ask natural-language questions — grounded in real filings — and receive cited, auditable answers.

The system is designed to run end-to-end in Docker Compose: Airflow orchestrates periodic ingestion from EDGAR, the FastAPI backend handles query routing and logging, PostgreSQL stores metadata and query logs, and the Streamlit frontend delivers a conversational interface with streaming responses.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Browser                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP / SSE
                    ┌───────────▼───────────┐
                    │   Streamlit Frontend  │  :8501
                    │  (chat + eval pages)  │
                    └───────────┬───────────┘
                                │ REST / SSE
                    ┌───────────▼───────────┐
                    │    FastAPI Backend    │  :8000
                    │  /query  /documents  │
                    │  /eval   /health     │
                    └─────┬──────────┬─────┘
                          │          │
             ┌────────────▼──┐  ┌────▼───────────┐
             │  PostgreSQL   │  │    Pinecone     │
             │  (metadata +  │  │  (vector index) │
             │   query logs) │  │                 │
             └────────────┬──┘  └─────────────────┘
                          │
             ┌────────────▼───────────────────┐
             │      Apache Airflow DAG        │  :8081
             │  edgar_ingestion → embed upsert│
             └────────────────────────────────┘
                          │
             ┌────────────▼──────────┐
             │    SEC EDGAR API      │
             │  (10-K / 10-Q filings)│
             └───────────────────────┘
```

**Data flow:**
1. Airflow DAG fetches filings from EDGAR and chunks text.
2. Chunks are embedded via OpenAI and upserted into Pinecone.
3. Metadata (ticker, period, status) is stored in PostgreSQL.
4. A user query enters the Streamlit chat → FastAPI → FinSightAgent.
5. Agent retrieves top-K chunks from Pinecone, calls Claude/GPT for synthesis.
6. Streamed answer + citations are sent back; query is logged to Postgres.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Docker Desktop** ≥ 24 | With Compose v2 |
| **OpenAI API key** | For embeddings (`text-embedding-3-small`) |
| **Anthropic API key** | For Claude answer synthesis |
| **Pinecone API key** | Free Serverless tier works |
| **AWS credentials** *(optional)* | For S3-backed filing storage |

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/yourorg/finsight.git
cd finsight
cp .env.example .env
# Edit .env and fill in: OPENAI_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# 2. Start all services
docker-compose up --build -d

# 3. Run Alembic migrations (first run only)
docker-compose exec backend alembic upgrade head

# 4. Trigger the ingestion DAG via Airflow UI
# Open http://localhost:8081  (admin / admin123)
# Enable "edgar_ingestion_dag" and trigger a run.
# Set Airflow Variable: EDGAR_TICKERS = '["AAPL","MSFT","GOOGL"]'

# 5. Open the Streamlit chat
# http://localhost:8501
```

---

## Evaluation Harness

FinSight includes a retrieval evaluation pipeline accessible from the Eval Dashboard.

| Metric | Description |
|---|---|
| **Hit Rate** | Fraction of queries where at least one expected chunk is retrieved |
| **MRR** | Mean Reciprocal Rank — rewards retrieving the expected chunk at a higher position |
| **Avg Score** | Average similarity score of retrieved chunks (0–1) |

The harness reads a JSONL eval dataset (default: `tests/fixtures/eval_dataset.jsonl`) where each line is:

```json
{"query": "What are AAPL revenue figures?", "expected_chunk_ids": ["aapl_10k_2023_chunk_5"]}
```

Run an evaluation from the dashboard UI or curl:
```bash
curl -X POST http://localhost:8000/eval/run \
  -H 'Content-Type: application/json' \
  -d '{"eval_dataset_path": "tests/fixtures/eval_dataset.jsonl"}'
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Ingestion** | Apache Airflow 2.10, httpx, BeautifulSoup4, LangChain text splitters |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Store** | Pinecone Serverless |
| **LLM** | Anthropic Claude 3 (Haiku/Sonnet) via `anthropic` SDK |
| **Backend API** | FastAPI, SQLAlchemy 2 async, Alembic, asyncpg |
| **Database** | PostgreSQL 15 |
| **Frontend** | Streamlit |
| **Infra** | Docker Compose |
| **Testing** | pytest, pytest-asyncio, httpx |

---

## Running Tests

```bash
# Unit tests (no live services needed)
docker-compose exec backend pytest tests/unit/ -v

# Integration tests (requires running backend + DB)
docker-compose exec backend pytest tests/integration/ -v

# End-to-end standalone script
python tests/integration/test_end_to_end.py
```
