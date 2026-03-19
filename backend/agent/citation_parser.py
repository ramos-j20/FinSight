"""Citation parser for the FinSight RAG agent."""
import re
from dataclasses import dataclass

from backend.embeddings.pinecone_client import RetrievedChunk


class CitationParseError(Exception):
    """Exception raised when citations cannot be parsed correctly."""
    pass


@dataclass
class Citation:
    """A citation mapping an LLM reference to a source chunk."""
    chunk_id: str
    ticker: str
    filing_type: str
    period: str
    excerpt: str
    reference_number: int
    
    def dict(self) -> dict:
        """Return dict representation of the citation."""
        from dataclasses import asdict
        return asdict(self)


def extract_citations(llm_response: str, retrieved_chunks: list[RetrievedChunk]) -> list[Citation]:
    """Extract citations from the LLM response and map them to retrieved chunks.
    
    Finds references like [1] or [2] in the text, maps them to the 1-indexed chunks,
    and returns a deduplicated list of Citation objects.
    
    Args:
        llm_response: The text response from the LLM.
        retrieved_chunks: The chunks that were provided to the LLM in the context.
        
    Returns:
        A deduplicated list of Citation objects.
        
    Raises:
        CitationParseError: If an invalid reference number is found (out of bounds).
    """
    # 1. Find all reference numbers
    matches = re.finditer(r'\[(\d+)\]', llm_response)
    
    citations = []
    seen_chunk_ids = set()
    
    for match in matches:
        ref_num = int(match.group(1))
        
        # 5. Validate bounds
        if ref_num < 1 or ref_num > len(retrieved_chunks):
            raise CitationParseError(
                f"Invalid reference number [{ref_num}]. Only {len(retrieved_chunks)} chunks were provided."
            )
            
        # 2. Map to chunk (1-indexed to 0-indexed)
        chunk = retrieved_chunks[ref_num - 1]
        
        # 4. Deduplicate by chunk_id
        if chunk.chunk_id in seen_chunk_ids:
            continue
            
        seen_chunk_ids.add(chunk.chunk_id)
        
        # 3. Build Citation object
        citation = Citation(
            chunk_id=chunk.chunk_id,
            ticker=chunk.ticker,
            filing_type=chunk.filing_type,
            period=chunk.period,
            excerpt=chunk.text[:200],
            reference_number=ref_num
        )
        citations.append(citation)
        
    return citations
