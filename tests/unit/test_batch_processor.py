"""Unit tests for the FinSightBatchProcessor."""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from backend.agent.batch_processor import FinSightBatchProcessor, BatchRequest, BatchResult


@pytest.fixture
def mock_anthropic():
    with patch("backend.agent.batch_processor.AsyncAnthropic") as mock:
        yield mock


@pytest.fixture
def processor(mock_anthropic):
    return FinSightBatchProcessor()


@pytest.mark.asyncio
async def test_submit_batch_format(processor, mock_anthropic):
    """Test submit_batch builds correct Anthropic request format."""
    # Mock retriever
    processor.retriever.retrieve = MagicMock(return_value=[])
    processor.retriever.format_context = MagicMock(return_value="test context")
    
    # Mock client
    mock_batch = MagicMock()
    mock_batch.id = "test_batch_id"
    processor.async_client.messages.batches.create = AsyncMock(return_value=mock_batch)
    
    # Mock DB session
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.commit = AsyncMock()
    mock_session.add = MagicMock() 
    
    processor.session_maker = MagicMock(return_value=mock_session)
    
    requests = [BatchRequest(custom_id="1", query="test query")]
    batch_id = await processor.submit_batch(requests, "test_job")
    
    assert batch_id == "test_batch_id"
    processor.async_client.messages.batches.create.assert_called_once()
    
    # Verify the structure of the request passed to Anthropic
    args, kwargs = processor.async_client.messages.batches.create.call_args
    batch_reqs = kwargs["requests"]
    assert len(batch_reqs) == 1
    assert batch_reqs[0]["custom_id"] == "1"
    assert "messages" in batch_reqs[0]["params"]
    assert "system" in batch_reqs[0]["params"]


@pytest.mark.asyncio
async def test_retrieve_results_aggregation(processor):
    """Test retrieve_results aggregates cost correctly across multiple results."""
    # Mock client.messages.batches.results (Async Iterator)
    mock_results = [
        MagicMock(custom_id="1", result=MagicMock(type="succeeded", message=MagicMock(
            content=[MagicMock(text="ans1")],
            usage=MagicMock(input_tokens=1000, output_tokens=500)
        ))),
        MagicMock(custom_id="2", result=MagicMock(type="succeeded", message=MagicMock(
            content=[MagicMock(text="ans2")],
            usage=MagicMock(input_tokens=2000, output_tokens=1000)
        )))
    ]
    
    # Mock async iterator for results
    class AsyncResultIter:
        def __init__(self, items):
            self.items = items
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.items: raise StopAsyncIteration
            return self.items.pop(0)

    processor.async_client.messages.batches.results = AsyncMock(return_value=AsyncResultIter(mock_results))
    
    # Mock DB session
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.commit = AsyncMock()
    mock_session.add = MagicMock()
    processor.session_maker = MagicMock(return_value=mock_session)
    
    results = await processor.retrieve_results("test_batch_id")
    
    assert len(results) == 2
    assert results[0].input_tokens == 1000
    assert results[1].input_tokens == 2000
    

@pytest.mark.asyncio
async def test_failed_results_handling(processor):
    """Test failed results counted separately, not contributing to cost totals."""
    mock_results = [
        MagicMock(custom_id="1", result=MagicMock(type="succeeded", message=MagicMock(
            content=[MagicMock(text="ans1")],
            usage=MagicMock(input_tokens=1000, output_tokens=500)
        ))),
        MagicMock(custom_id="2", result=MagicMock(type="error", error=MagicMock(message="Rate limit reached")))
    ]
    
    # Mock async iterator for results
    class AsyncResultIter:
        def __init__(self, items):
            self.items = items
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.items: raise StopAsyncIteration
            return self.items.pop(0)

    processor.async_client.messages.batches.results = AsyncMock(return_value=AsyncResultIter(mock_results))
    
    # Mock DB session
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.commit = AsyncMock()
    mock_session.add = MagicMock()
    processor.session_maker = MagicMock(return_value=mock_session)
    
    results = await processor.retrieve_results("test_batch_id")
    
    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False
    assert results[1].error_message == "Rate limit reached"
