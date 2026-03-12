"""API routes for evaluation operations."""
from typing import Any

from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def run_eval() -> dict[str, Any]:
    return {"message": "Eval endpoint stub"}
