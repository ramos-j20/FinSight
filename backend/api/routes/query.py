"""API routes for query operations."""
from typing import Any

from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def query_endpoint() -> dict[str, Any]:
    return {"message": "Query endpoint stub"}
