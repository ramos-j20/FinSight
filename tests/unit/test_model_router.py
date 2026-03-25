import pytest
from backend.agent.model_router import resolve_model, detect_query_complexity, MODE_CONFIG, RoutingDecision

def test_auto_detect_comparison():
    mode, reason = detect_query_complexity("How did Q1 2023 revenue compare to Q1 2022?", "comparison")
    assert mode == "deep"
    assert reason == "comparison detected"

def test_auto_detect_keyword_risk():
    mode, reason = detect_query_complexity("What are the primary risk factors for the company?", "rag")
    assert mode == "deep"
    assert reason == "keyword: risk"

def test_auto_detect_generic_fast():
    mode, reason = detect_query_complexity("What is the company name?", "rag")
    assert mode == "fast"
    assert reason == "default: fast"

def test_resolve_auto_comparison():
    decision: RoutingDecision = resolve_model(mode_override="auto", query="Apple vs Microsoft revenue", query_type="comparison")
    assert decision.mode_used == "deep"
    assert decision.routing_reason == "comparison detected"
    assert decision.model == MODE_CONFIG["deep"]["model"]
    assert decision.top_k == MODE_CONFIG["deep"]["top_k"]
    assert decision.max_tokens == MODE_CONFIG["deep"]["max_tokens"]

def test_resolve_auto_keyword_risk():
    decision: RoutingDecision = resolve_model(mode_override=None, query="Explain the litigation history.", query_type="rag")
    assert decision.mode_used == "deep"
    assert decision.routing_reason == "keyword: litigation"
    assert decision.model == MODE_CONFIG["deep"]["model"]
    assert decision.top_k == MODE_CONFIG["deep"]["top_k"]
    assert decision.max_tokens == MODE_CONFIG["deep"]["max_tokens"]

def test_resolve_auto_generic():
    decision: RoutingDecision = resolve_model(mode_override=None, query="Tell me about the CEO.", query_type="rag")
    assert decision.mode_used == "fast"
    assert decision.routing_reason == "default: fast"
    assert decision.model == MODE_CONFIG["fast"]["model"]

def test_resolve_override_fast():
    decision: RoutingDecision = resolve_model(mode_override="fast", query="What are the risk factors?", query_type="rag")
    # Even though query has "risk" which triggers deep in auto mode, override should bypass it.
    assert decision.mode_used == "fast"
    assert decision.routing_reason == "user override"
    assert decision.model == MODE_CONFIG["fast"]["model"]

def test_resolve_override_deep():
    decision: RoutingDecision = resolve_model(mode_override="deep", query="Simple question.", query_type="rag")
    assert decision.mode_used == "deep"
    assert decision.routing_reason == "user override"
    assert decision.model == MODE_CONFIG["deep"]["model"]

def test_resolve_override_invalid():
    # Test that an invalid override falls back to "fast" safely
    decision: RoutingDecision = resolve_model(mode_override="invalid_mode", query="Simple question.", query_type="rag")
    assert decision.mode_used == "fast"
    assert decision.routing_reason == "user override"
    assert decision.model == MODE_CONFIG["fast"]["model"]
