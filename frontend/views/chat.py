"""Chat page — multi-turn RAG chat with streaming and citation cards."""
import json
import time
import uuid

import httpx
import streamlit as st

from frontend.components.citation_card import render_citation_card
from frontend.components.filing_selector import render_filing_selector


def _init_session_state() -> None:
    """Initialise session state keys on first run."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "query_log_ids" not in st.session_state:
        st.session_state["query_log_ids"] = {}  # message_index -> query_log_id
    if "feedback_submitted" not in st.session_state:
        st.session_state["feedback_submitted"] = set()


def _stream_query(
    backend_url: str,
    query: str,
    session_id: str,
    ticker_filter: str | None,
    filing_type_filter: str | None,
    conversation_history: list[dict],
    mode_override: str | None,
) -> tuple[str, list[dict], int, str, int | None, str, str, str]:
    """Call POST /query/stream and collect streamed tokens.

    Returns:
        (full_answer, citations, latency_ms, prompt_version, query_log_id, model_used, mode_used, routing_reason)
    """
    payload: dict = {
        "query": query,
        "session_id": session_id,
        "conversation_history": conversation_history,
        "mode_override": mode_override,
    }
    if ticker_filter:
        payload["ticker_filter"] = ticker_filter
    if filing_type_filter:
        payload["filing_type_filter"] = filing_type_filter

    answer_parts: list[str] = []
    citations: list[dict] = []
    latency_ms: int = 0
    prompt_version: str = "v1"
    query_log_id: int | None = None
    model_used: str = ""
    mode_used: str = ""
    routing_reason: str = ""

    start = time.monotonic()

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{backend_url}/query/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[5:].strip()
                if not data_str:
                    continue
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                chunk_type = chunk.get("type", "")
                if chunk_type == "text":
                    answer_parts.append(chunk.get("data", ""))
                elif chunk_type == "citations":
                    raw = chunk.get("data", {})
                    if isinstance(raw, dict):
                        citations = raw.get("citations", [])
                        latency_ms = raw.get("latency_ms", 0)
                        prompt_version = raw.get("prompt_version", "v1")
                        query_log_id = raw.get("query_log_id")
                        model_used = raw.get("model_used", "")
                        mode_used = raw.get("mode_used", "")
                        routing_reason = raw.get("routing_reason", "")
                elif chunk_type == "error":
                    raise RuntimeError(chunk.get("data", "Unknown streaming error"))

    if not latency_ms:
        latency_ms = int((time.monotonic() - start) * 1000)

    return "".join(answer_parts), citations, latency_ms, prompt_version, query_log_id, model_used, mode_used, routing_reason


def _submit_feedback(backend_url: str, query_log_id: int, score: int) -> None:
    """POST /query/{id}/feedback."""
    try:
        httpx.post(
            f"{backend_url}/query/{query_log_id}/feedback",
            json={"query_log_id": query_log_id, "score": score},
            timeout=10,
        )
    except Exception as exc:
        st.warning(f"Feedback submission failed: {exc}")


def render_chat_page(backend_url: str) -> None:
    """Render the full chat page."""
    _init_session_state()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem; font-weight: 800; margin-bottom: 0;
        ">💬 FinSight Chat</h1>
        <p style="color: #94a3b8; margin-top: 4px;">
            Ask any question about SEC filings. Responses are grounded in indexed documents.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Filing selector ───────────────────────────────────────────────────────
    with st.container():
        ticker_filter, filing_type_filter = render_filing_selector(backend_url)

    st.markdown("---")
    
    mode = st.radio(
        "Query Mode",
        ["🤖 Auto", "⚡ Fast", "🔬 Deep"],
        horizontal=True,
        help="Auto selects the best model based on your query. Fast = Haiku. Deep = Sonnet."
    )

    mode_map = {
        "🤖 Auto": "auto",
        "⚡ Fast": "fast",
        "🔬 Deep": "deep"
    }

    st.markdown("---")

    # ── Control buttons ───────────────────────────────────────────────────────
    btn_col1, btn_col2, _ = st.columns([1, 1, 5])
    with btn_col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["query_log_ids"] = {}
            st.session_state["feedback_submitted"] = set()
            st.rerun()
    with btn_col2:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["query_log_ids"] = {}
            st.session_state["feedback_submitted"] = set()
            st.rerun()

    # ── Message history ───────────────────────────────────────────────────────
    messages = st.session_state["messages"]
    for idx, msg in enumerate(messages):
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if role == "assistant":
                # Render citations if any
                for cit in msg.get("citations", []):
                    render_citation_card(cit)
                # Latency / version footer
                if msg.get("latency_ms") or msg.get("prompt_version"):
                    model = msg.get("model_used", "claude-haiku")
                    md = msg.get("mode_used", "fast")
                    reason = msg.get("routing_reason", "default")
                    md_display = f"{md} (auto-routed)" if reason != "user override" else md
                    st.caption(
                        f"Model: {model}  |  Mode: {md_display}  |  Reason: {reason}  |  Latency: {msg.get('latency_ms', 0)}ms"
                    )
                # Feedback widget
                qlog_id = st.session_state["query_log_ids"].get(idx)
                if qlog_id and idx not in st.session_state["feedback_submitted"]:
                    fb_key = f"feedback_{idx}"
                    score = st.feedback("stars", key=fb_key)
                    if score is not None:
                        _submit_feedback(backend_url, qlog_id, score + 1)  # 0-indexed → 1-5
                        st.session_state["feedback_submitted"].add(idx)
                        st.success("Thanks for your feedback! ⭐")

    # ── Query input ───────────────────────────────────────────────────────────
    query = st.chat_input("Ask about any SEC filing…", key="chat_input")

    if query:
        # Add user message
        messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Build conversation history (last 10 turns)
        convo_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[:-1][-10:]
        ]

        # Stream assistant response
        with st.chat_message("assistant"):
            response_container = st.empty()
            try:
                answer, citations, latency_ms, prompt_version, query_log_id, model_used, mode_used, routing_reason = _stream_query(
                    backend_url=backend_url,
                    query=query,
                    session_id=st.session_state["session_id"],
                    ticker_filter=ticker_filter,
                    filing_type_filter=filing_type_filter,
                    conversation_history=convo_history,
                    mode_override=mode_map[mode],
                )
                response_container.markdown(answer)

                # Render citations
                if citations:
                    st.markdown("**📚 Sources:**")
                    for cit in citations:
                        render_citation_card(cit)

                # Latency footer
                md_display = f"{mode_used} (auto-routed)" if routing_reason != "user override" else mode_used
                st.caption(
                    f"Model: {model_used}  |  Mode: {md_display}  |  Reason: {routing_reason}  |  Latency: {latency_ms}ms"
                )

                # Persist to session state
                assistant_idx = len(messages)
                messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "latency_ms": latency_ms,
                    "prompt_version": prompt_version,
                    "model_used": model_used,
                    "mode_used": mode_used,
                    "routing_reason": routing_reason,
                })
                if query_log_id:
                    st.session_state["query_log_ids"][assistant_idx] = query_log_id

                # Inline feedback for the brand new message
                if query_log_id:
                    score = st.feedback("stars", key=f"feedback_{assistant_idx}")
                    if score is not None:
                        _submit_feedback(backend_url, query_log_id, score + 1)
                        st.session_state["feedback_submitted"].add(assistant_idx)
                        st.success("Thanks for your feedback! ⭐")

            except Exception as exc:
                error_text = f"❌ Error: {exc}"
                response_container.error(error_text)
                messages.append({"role": "assistant", "content": error_text, "citations": []})

        st.session_state["messages"] = messages
