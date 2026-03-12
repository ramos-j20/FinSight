"""Custom exception classes for FinSight."""

class FinSightBaseException(Exception):
    """Base exception class for all custom FinSight exceptions."""
    pass


class EDGARFetchError(FinSightBaseException):
    """Raised when fetching documents from the SEC EDGAR API fails."""
    pass


class S3UploadError(FinSightBaseException):
    """Raised when uploading or downloading files to/from AWS S3 fails."""
    pass


class EmbeddingError(FinSightBaseException):
    """Raised when generating embeddings via OpenAI API fails."""
    pass


class PineconeUpsertError(FinSightBaseException):
    """Raised when upserting to or querying Pinecone fails."""
    pass


class AgentError(FinSightBaseException):
    """Raised when the custom RAG agent orchestration loop fails."""
    pass


class CitationParseError(FinSightBaseException):
    """Raised when the expected citation structure cannot be parsed from the LLM output."""
    pass
