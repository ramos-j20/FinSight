"""Retriever module for FinSight agent."""
from backend.embeddings.pinecone_client import PineconeClient, RetrievedChunk
from backend.embeddings.embedder import OpenAIEmbedder
from backend.core.logging import get_logger

logger = get_logger(__name__)

class FinSightRetriever:
    """Retriever for FinSight SEC filings chunks."""
    
    def __init__(self) -> None:
        self._embedder = OpenAIEmbedder()
        self._pinecone = PineconeClient()
        logger.info("FinSightRetriever initialised")
        
    def retrieve(
        self, 
        query: str, 
        top_k: int, 
        ticker_filter: str | None = None, 
        filing_type_filter: str | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks from Pinecone.
        
        Args:
            query: The user's query.
            top_k: Number of results to return.
            ticker_filter: Optional ticker to filter by.
            filing_type_filter: Optional filing type to filter by.
            
        Returns:
            List of RetrievedChunk objects ordered by score descending.
        """
        embedding = self._embedder.embed_text(query)
        
        filter_dict = {}
        if ticker_filter:
            filter_dict["ticker"] = ticker_filter
        if filing_type_filter:
            filter_dict["filing_type"] = filing_type_filter
            
        pinecone_filter = filter_dict if filter_dict else None
        
        chunks = self._pinecone.query(
            embedding=embedding,
            top_k=top_k,
            filter=pinecone_filter
        )
        
        # Sort chunks by score descending just to be safe, though Pinecone typically sorts them.
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(
            "Retrieved chunks", 
            query=query, 
            top_k=top_k, 
            retrieved_count=len(chunks)
        )
        return chunks[:top_k]
        
    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context string.
        
        Args:
            chunks: List of RetrievedChunk objects.
            
        Returns:
            Formatted context string.
        """
        if not chunks:
            return "No context found."
            
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source_header = f"[{i}] Source: {chunk.ticker} | {chunk.filing_type} | {chunk.period}"
            context_parts.append(f"{source_header}\n{chunk.text}")
            
        return "\n---\n".join(context_parts)
