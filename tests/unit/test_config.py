"""Unit tests for the application configuration settings."""
import os
from unittest import mock

from backend.core.config import Settings, get_settings


@mock.patch.dict(os.environ, {
    "ANTHROPIC_API_KEY": "test-anthropic",
    "OPENAI_API_KEY": "test-openai",
    "PINECONE_API_KEY": "test-pinecone",
    "PINECONE_INDEX_NAME": "test-index",
    "PINECONE_ENVIRONMENT": "us-east-1",
    "AWS_ACCESS_KEY_ID": "test-aws-key",
    "AWS_SECRET_ACCESS_KEY": "test-aws-secret",
    "AWS_S3_BUCKET_NAME": "test-bucket",
    "AWS_REGION": "us-east-1",
    "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/db",
    "EDGAR_USER_AGENT": "Test User test@example.com",
})
def test_get_settings_loads_env_vars():
    """Test that settings load correctly from the environment."""
    # Clear cache to force a fresh reload from our mocked environ
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.ANTHROPIC_API_KEY == "test-anthropic"
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@localhost/db"
    assert settings.LOG_LEVEL == "INFO"
    assert settings.CHUNK_SIZE == 1000
