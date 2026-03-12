"""Database session management and engine configuration."""
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import get_settings


# Create module-level engine and sessionmaker placeholders
engine: AsyncEngine | None = None
async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Retrieve or create the async SQLAlchemy engine."""
    global engine
    if engine is None:
        settings = get_settings()
        # Ensure URLs are properly casted for asyncpg
        url = settings.DATABASE_URL
        if url and url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        engine = create_async_engine(url, echo=False)
    return engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Retrieve or create the async session factory."""
    global async_session_maker
    if async_session_maker is None:
        async_engine = get_engine()
        async_session_maker = async_sessionmaker(
            bind=async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async database session.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
