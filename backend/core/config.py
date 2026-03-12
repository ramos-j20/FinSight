"""Configuration settings for the FinSight application."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: str
    OPENAI_API_KEY: str
    
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    PINECONE_ENVIRONMENT: str
    
    # AWS S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_S3_BUCKET_NAME: str
    AWS_REGION: str
    
    # Database
    DATABASE_URL: str
    
    # SEC / EDGAR
    EDGAR_USER_AGENT: str
    
    # System settings
    LOG_LEVEL: str = "INFO"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_TOP_K: int = 10
    PROMPT_VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    Uses lru_cache to implement a singleton pattern.
    """
    return Settings()
