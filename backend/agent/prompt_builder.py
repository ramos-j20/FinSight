"""Prompts for FinSight RAG agent."""

PROMPT_VERSION = "1.0.0"

def build_system_prompt(enable_caching: bool = False) -> str | list[dict]:
    """Build the system prompt for the FinSight agent."""
    text = """You are an expert financial document analyst specializing in SEC filings.
Your task is to answer user queries based ONLY on the provided context retrieved from SEC filings.

CRITICAL RULES:
1. Answer ONLY based on the provided context. Never hallucinate financial data or information not present in the context.
2. If the answer cannot be found in the context, say so explicitly.
3. Always cite sources by their reference number (e.g., [1], [2]) when making points. Do not provide links or arbitrary URLs.
4. For comparative questions (e.g., "Q3 2024 vs Q3 2023"), structure your answer with clear before/after sections.
5. If the user requests structured JSON output, return a valid JSON object as instructed.
"""
    if enable_caching:
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    return text


def build_rag_prompt(
    query: str, 
    context: str, 
    conversation_history: list[dict], 
    enable_caching: bool = True
) -> list[dict]:
    """Build the messages array for a standard RAG query.
    
    Args:
        query: The user's query.
        context: The formatted context string.
        conversation_history: List of previous conversation turns (dict with 'role' and 'content').
        enable_caching: Whether to enable prompt caching for context.
        
    Returns:
        List of dictionaries formatted for the Anthropic API.
    """
    messages = list(conversation_history)
    
    context_content = []
    
    # Estimate tokens: len(context) // 4 > 1024
    if enable_caching and (len(context) // 4 > 1024):
        context_content.append({
            "type": "text",
            "text": f"<context>\n{context}\n</context>",
            "cache_control": {"type": "ephemeral"}
        })
    else:
        context_content.append({
            "type": "text",
            "text": f"<context>\n{context}\n</context>"
        })
        
    context_content.append({
        "type": "text",
        "text": f"\n\nQuestion: {query}\n\nProvide a detailed answer with citations to the numbered sources above."
    })

    messages.append({
        "role": "user",
        "content": context_content
    })
    
    return messages


def build_comparison_prompt(
    query: str, 
    context: str, 
    ticker: str, 
    periods: list[str],
    enable_caching: bool = True
) -> list[dict]:
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
    
    context_content = []
    if enable_caching and (len(context) // 4 > 1024):
        context_content.append({
            "type": "text",
            "text": f"<context>\n{context}\n</context>",
            "cache_control": {"type": "ephemeral"}
        })
    else:
        context_content.append({
            "type": "text",
            "text": f"<context>\n{context}\n</context>"
        })

    context_content.append({
        "type": "text",
        "text": f"\n\nQuestion: {query}\n\nProvide a comprehensive comparison for {ticker} across the following periods: {periods_str}.\n\nStructure your answer with:\n1. A high-level **Summary** explaining the main trends.\n2. A detailed **Period Analysis** comparing the specific metrics for each period.\n3. Clear **Key Drivers** explaining why the changes occurred.\n\nIMPORTANT: Use citations in the format [1], [2], etc., whenever you reference data from the context. Ensure your final answer is nicely formatted in MarkDown for a chat interface."
    })

    return [
        {
            "role": "user",
            "content": context_content
        }
    ]
