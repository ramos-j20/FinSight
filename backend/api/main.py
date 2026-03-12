"""Main FastAPI application entrypoint."""
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import documents, eval, query
from backend.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events."""
    logger.info("FinSight API starting up...")
    yield
    logger.info("FinSight API shutting down...")


app = FastAPI(
    title="FinSight API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return the health status of the API."""
    return {"status": "ok", "version": "0.1.0"}


app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(eval.router, prefix="/eval", tags=["eval"])
