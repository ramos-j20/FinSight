"""API routes for document management."""
from typing import Any

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_documents() -> dict[str, Any]:
    return {"message": "Documents endpoint stub"}
