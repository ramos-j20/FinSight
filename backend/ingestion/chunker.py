"""Document chunker — splits parsed filings into overlapping chunks."""
import json
from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.ingestion.parser import ParsedDocument

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of a parsed filing document."""

    chunk_id: str
    ticker: str
    filing_type: str
    period: str
    text: str
    char_count: int
    chunk_index: int
    total_chunks: int


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Split a ParsedDocument into overlapping text chunks.

    Uses LangChain's RecursiveCharacterTextSplitter for intelligent
    splitting that respects paragraph and sentence boundaries.

    Chunk IDs are deterministic:  {ticker}_{filing_type}_{period}_{chunk_index}
    This enables idempotent re-indexing (same input → same chunk IDs).

    Args:
        doc: The parsed document to split.
        chunk_size: Target size in characters for each chunk (default from settings).
        chunk_overlap: Number of overlapping characters between chunks (default from settings).

    Returns:
        List of DocumentChunk objects.
    """
    settings = get_settings()
    actual_chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    actual_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    logger.info(
        "Chunking document",
        ticker=doc.ticker,
        filing_type=doc.filing_type,
        period=doc.period,
        char_count=doc.char_count,
        chunk_size=actual_chunk_size,
        chunk_overlap=actual_chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=actual_chunk_size,
        chunk_overlap=actual_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts = splitter.split_text(doc.clean_text)
    total = len(texts)

    chunks: list[DocumentChunk] = []
    for idx, text in enumerate(texts):
        chunk_id = f"{doc.ticker}_{doc.filing_type}_{doc.period}_{idx}"
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                ticker=doc.ticker,
                filing_type=doc.filing_type,
                period=doc.period,
                text=text,
                char_count=len(text),
                chunk_index=idx,
                total_chunks=total,
            )
        )

    logger.info(
        "Document chunked",
        ticker=doc.ticker,
        total_chunks=total,
    )
    return chunks


def serialize_chunks(chunks: list[DocumentChunk]) -> str:
    """Serialize a list of DocumentChunks to a JSON string for S3 storage.

    Args:
        chunks: The list of chunks to serialize.

    Returns:
        A JSON string representation of the chunks.
    """
    return json.dumps([asdict(c) for c in chunks], indent=2)


def deserialize_chunks(json_str: str) -> list[DocumentChunk]:
    """Deserialize a JSON string back into a list of DocumentChunk objects.

    Args:
        json_str: JSON string previously produced by serialize_chunks.

    Returns:
        A list of DocumentChunk objects.
    """
    data = json.loads(json_str)
    return [DocumentChunk(**item) for item in data]
