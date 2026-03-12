# FinSight: Earnings Intelligence Platform

An end-to-end AI platform that ingests SEC EDGAR filings and earnings call transcripts, indexes them into a vector store, and exposes a multi-turn RAG agent capable of answering complex, comparative financial questions with source citations.

## Tech Stack

| Layer                      | Technology                          |
|----------------------------|-------------------------------------|
| Pipeline Orchestration     | Apache Airflow 2.x                  |
| Raw Data Storage           | AWS S3 (boto3)                      |
| Document Parsing           | LangChain document loaders          |
| Chunking                   | LangChain text splitters            |
| Embeddings                 | OpenAI text-embedding-3-small       |
| Vector Store               | Pinecone (serverless)               |
| LLM                        | Anthropic Claude (claude-sonnet-4-6)|
| Agent Orchestration        | Custom Python (NO LangChain agents) |
| Backend API                | FastAPI (async)                     |
| Frontend / Dashboard       | Streamlit                           |
| Metadata Store             | PostgreSQL                          |
| Containerization           | Docker + Docker Compose             |
| Config Management          | python-dotenv + pydantic Settings   |
| Testing                    | pytest + pytest-asyncio             |
