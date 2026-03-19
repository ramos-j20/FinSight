import pytest
from backend.agent.citation_parser import extract_citations, CitationParseError
from backend.embeddings.pinecone_client import RetrievedChunk

@pytest.fixture
def sample_chunks():
    return [
        RetrievedChunk(chunk_id="c1", ticker="AAPL", filing_type="10-K", period="2024", text="Apple makes phones.", score=0.9, source_url=""),
        RetrievedChunk(chunk_id="c2", ticker="AAPL", filing_type="10-K", period="2024", text="Apple makes tablets.", score=0.8, source_url=""),
        RetrievedChunk(chunk_id="c3", ticker="MSFT", filing_type="10-K", period="2024", text="Microsoft makes software.", score=0.7, source_url="")
    ]

def test_extract_citations_success(sample_chunks):
    response = "Apple relies on iPhone sales [1]. They also make iPads [2]."
    citations = extract_citations(response, sample_chunks)
    
    assert len(citations) == 2
    assert citations[0].chunk_id == "c1"
    assert citations[0].reference_number == 1
    assert citations[1].chunk_id == "c2"
    assert citations[1].reference_number == 2

def test_extract_citations_deduplication(sample_chunks):
    response = "Apple makes phones [1]. Also phones [1]."
    citations = extract_citations(response, sample_chunks)
    assert len(citations) == 1

def test_extract_citations_out_of_bounds(sample_chunks):
    response = "This is a fact [99]."
    with pytest.raises(CitationParseError):
        extract_citations(response, sample_chunks)
