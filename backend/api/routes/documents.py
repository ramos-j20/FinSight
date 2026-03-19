"""API routes for document management."""
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.models import DocumentListResponse, FilingMetadataResponse
from backend.db.models import FilingMetadata
from backend.db.session import get_db

router = APIRouter()


def _to_response(filing: FilingMetadata) -> FilingMetadataResponse:
    """Convert a FilingMetadata ORM row to a FilingMetadataResponse."""
    return FilingMetadataResponse(
        id=filing.id,
        ticker=filing.ticker,
        filing_type=filing.filing_type,
        period=filing.period,
        status=filing.status.value if hasattr(filing.status, "value") else str(filing.status),
        chunk_count=filing.chunk_count,
        is_embedded=filing.is_embedded,
        ingested_at=filing.ingested_at.isoformat() if filing.ingested_at else "",
    )


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


@router.get("/", response_model=DocumentListResponse)
async def get_documents(
    ticker: str | None = Query(default=None),
    filing_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Return a paginated list of filing metadata records."""
    stmt = select(FilingMetadata)
    count_stmt = select(func.count()).select_from(FilingMetadata)

    if ticker:
        stmt = stmt.where(FilingMetadata.ticker == ticker.upper())
        count_stmt = count_stmt.where(FilingMetadata.ticker == ticker.upper())
    if filing_type:
        stmt = stmt.where(FilingMetadata.filing_type == filing_type)
        count_stmt = count_stmt.where(FilingMetadata.filing_type == filing_type)
    if status:
        stmt = stmt.where(FilingMetadata.status == status.upper())
        count_stmt = count_stmt.where(FilingMetadata.status == status.upper())

    total_count_result = await db.execute(count_stmt)
    total_count: int = total_count_result.scalar_one()

    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)

    result = await db.execute(stmt)
    filings = result.scalars().all()

    return DocumentListResponse(
        filings=[_to_response(f) for f in filings],
        total_count=total_count,
    )


# ---------------------------------------------------------------------------
# GET /documents/{ticker}/filings
# ---------------------------------------------------------------------------


@router.get("/{ticker}/filings", response_model=list[FilingMetadataResponse])
async def get_ticker_filings(
    ticker: str,
    db: AsyncSession = Depends(get_db),
) -> list[Any]:
    """Return all filings for a specific ticker."""
    stmt = select(FilingMetadata).where(FilingMetadata.ticker == ticker.upper())
    result = await db.execute(stmt)
    filings = result.scalars().all()
    return [_to_response(f) for f in filings]
