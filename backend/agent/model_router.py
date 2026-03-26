"""Model routing logic for choosing between fast and deep reasoning."""
import re
from typing import Literal
from pydantic import BaseModel

class ModelRoute(BaseModel):
    """Result of a model routing decision."""
    model: str
    mode_used: str
    routing_reason: str
    max_tokens: int = 1000
    top_k: int = 10

# Config for different modes
MODE_CONFIG = {
    "fast": "claude-haiku-4-5",
    "deep": "claude-sonnet-4-5"
}

def detect_query_complexity(query: str) -> Literal["fast", "deep"]:
    """Detect if a query requires deep reasoning or just fast retrieval."""
    complex_keywords = [
        r"analyze", r"compare", r"trend", r"why", r"impact",
        r"risk", r"forecast", r"strategy", r"sentiment",
        r"comprehensive", r"deep", r"investigate",
        r"earnings", r"eps", r"revenue", r"financials",
        r"balance sheet", r"cash flow", r"10-k", r"10-q"
    ]
    
    # Check for keyword matches
    if any(re.search(kw, query, re.IGNORECASE) for kw in complex_keywords):
        return "deep"
    
    # Large queries suggest multiple topics or complex intent
    if len(query.split()) > 15:
        return "deep"
    
    return "fast"

def resolve_model(query: str, mode_override: str | None = None, query_type: str | None = None) -> ModelRoute:
    """Determine which model and mode to use for a given query."""
    # Determine mode
    if mode_override and mode_override in MODE_CONFIG:
        mode = mode_override
        reason = f"Explicit override to '{mode}' mode."
    else:
        mode = detect_query_complexity(query)
        reason = f"Complexity detection routed to '{mode}' mode."
    
    model = MODE_CONFIG[mode]
    
    # Set parameters based on query type or mode
    max_tokens = 2000 if mode == "deep" or query_type == "comparison" else 1000
    top_k = 150 if mode == "deep" or query_type == "comparison" else 75
    
    return ModelRoute(
        model=model,
        mode_used=mode,
        routing_reason=reason,
        max_tokens=max_tokens,
        top_k=top_k
    )
