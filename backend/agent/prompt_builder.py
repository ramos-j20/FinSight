"""Prompts for FinSight RAG agent."""

PROMPT_VERSION = "1.0.0"

def build_system_prompt() -> str:
    """Build the system prompt for the FinSight agent."""
    return """You are an expert financial document analyst specializing in SEC filings.
Your task is to answer user queries based ONLY on the provided context retrieved from SEC filings.

CRITICAL RULES:
1. Answer ONLY based on the provided context. Never hallucinate financial data or information not present in the context.
2. If the answer cannot be found in the context, say so explicitly.
3. Always cite sources by their reference number (e.g., [1], [2]) when making points. Do not provide links or arbitrary URLs.
4. For comparative questions (e.g., "Q3 2024 vs Q3 2023"), structure your answer with clear before/after sections.
5. If the user requests structured JSON output, return a valid JSON object as instructed.
"""

def build_rag_prompt(query: str, context: str, conversation_history: list[dict]) -> list[dict]:
    """Build the messages array for a standard RAG query.
    
    Args:
        query: The user's query.
        context: The formatted context string.
        conversation_history: List of previous conversation turns (dict with 'role' and 'content').
        
    Returns:
        List of dictionaries formatted for the Anthropic API.
    """
    messages = list(conversation_history)
    
    user_content = f"""<context>
{context}
</context>

Question: {query}

Provide a detailed answer with citations to the numbered sources above."""

    messages.append({
        "role": "user",
        "content": user_content
    })
    
    return messages

def build_comparison_prompt(query: str, context: str, ticker: str, periods: list[str]) -> list[dict]:
    """Build the messages array for a comparative query.
    
    Instructs the model to return a structured JSON response.
    
    Args:
        query: The user's query.
        context: The formatted context string.
        ticker: The stock ticker being compared.
        periods: List of periods being compared (e.g., ['Q3 2023', 'Q3 2024']).
        
    Returns:
        List of dictionaries formatted for the Anthropic API.
    """
    periods_str = ", ".join(periods)
    user_content = f"""<context>
{context}
</context>

Question: {query}

You must return a structured JSON response containing the comparison for {ticker} across the following periods: {periods_str}.
Do NOT output any markdown blocks (e.g., ```json) around your response, just return the raw JSON object.

The JSON should have exactly this format:
{{
  "summary": "your high-level summary of the comparison",
  "period_1": {{
    "period": "first period name",
    "key_points": ["point 1", "point 2"]
  }},
  "period_2": {{
    "period": "second period name",
    "key_points": ["point 1", "point 2"]
  }},
  "citations": [
    {{
      "reference_number": 1,
      "text": "brief explanation of what this source supports"
    }}
  ]
}}

Ensure all citations reference the numbered sources provided in the context.
"""

    return [
        {
            "role": "user",
            "content": user_content
        }
    ]
