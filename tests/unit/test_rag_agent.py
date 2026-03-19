import pytest
from unittest.mock import MagicMock, patch

from backend.agent.rag_agent import FinSightAgent, QueryRequest, AgentResponse
from backend.embeddings.pinecone_client import RetrievedChunk

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1", 
            ticker="AAPL", 
            filing_type="10-K", 
            period="2024", 
            text="Context 1", 
            score=0.9, 
            source_url=""
        )
    ]
    retriever.format_context.return_value = "[1] Source: AAPL | 10-K | 2024\nContext 1"
    return retriever

@pytest.fixture
def mock_anthropic():
    client = MagicMock()
    mock_msg = MagicMock()
    # Mock response.content[0].text
    mock_block = MagicMock()
    mock_block.text = "Here is the answer [1]."
    mock_msg.content = [mock_block]
    client.messages.create.return_value = mock_msg
    return client

@patch("backend.agent.rag_agent.Anthropic")
@patch("backend.agent.rag_agent.AsyncAnthropic")
@patch("backend.agent.rag_agent.FinSightRetriever")
def test_query_standard(mock_retriever_class, mock_async_anthropic_class, mock_anthropic_class, mock_retriever, mock_anthropic):
    mock_retriever_class.return_value = mock_retriever
    mock_anthropic_class.return_value = mock_anthropic
    
    agent = FinSightAgent()
    req = QueryRequest(query="What is AAPL?", session_id="123")
    
    response = agent.query(req)
    
    mock_retriever.retrieve.assert_called_once()
    mock_anthropic.messages.create.assert_called_once()
    assert isinstance(response, AgentResponse)
    assert response.answer == "Here is the answer [1]."
    assert len(response.citations) == 1
    assert response.latency_ms >= 0

@patch("backend.agent.rag_agent.Anthropic")
@patch("backend.agent.rag_agent.AsyncAnthropic")
@patch("backend.agent.rag_agent.FinSightRetriever")
def test_query_comparative(mock_retriever_class, mock_async_anthropic_class, mock_anthropic_class, mock_retriever, mock_anthropic):
    mock_retriever_class.return_value = mock_retriever
    mock_anthropic_class.return_value = mock_anthropic
    
    agent = FinSightAgent()
    req = QueryRequest(query="AAPL vs MSFT", session_id="123")
    
    response = agent.query(req)
    
    call_args = mock_anthropic.messages.create.call_args[1]
    messages = call_args["messages"]
    assert len(messages) == 1
    assert "structured JSON response" in messages[0]["content"]

# Async tests could be added but the logic is analogous
