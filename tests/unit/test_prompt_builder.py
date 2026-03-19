import pytest
from backend.agent.prompt_builder import (
    build_system_prompt,
    build_rag_prompt,
    build_comparison_prompt,
    PROMPT_VERSION
)

def test_prompt_version_exists():
    assert PROMPT_VERSION == "1.0.0"

def test_build_system_prompt():
    prompt = build_system_prompt()
    assert "expert financial document analyst" in prompt
    assert "CRITICAL RULES" in prompt

def test_build_rag_prompt():
    conversation = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    messages = build_rag_prompt("AAPL risk", "Context text", conversation)
    
    assert len(messages) == 3
    assert messages[0] == conversation[0]
    assert messages[1] == conversation[1]
    
    last_msg = messages[2]
    assert last_msg["role"] == "user"
    assert "Context text" in last_msg["content"]
    assert "AAPL risk" in last_msg["content"]

def test_build_comparison_prompt():
    messages = build_comparison_prompt("AAPL vs MSFT", "Context text", "AAPL", ["Q3 2023", "Q4 2023"])
    assert len(messages) == 1
    content = messages[0]["content"]
    assert "Context text" in content
    assert "AAPL vs MSFT" in content
    assert "Q3 2023, Q4 2023" in content
    assert "structured JSON response" in content
