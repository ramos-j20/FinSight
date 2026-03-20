"""Citation card component — renders a single retrieval citation."""
import streamlit as st


def render_citation_card(citation: dict) -> None:
    """Render a single citation inside a styled Streamlit expander.

    Args:
        citation: Dict with keys: reference_number, ticker, filing_type,
                  period, excerpt, chunk_id.
    """
    ref = citation.get("reference_number", "?")
    ticker = citation.get("ticker", "N/A")
    filing_type = citation.get("filing_type", "N/A")
    period = citation.get("period", "N/A")
    excerpt = citation.get("excerpt", "")
    chunk_id = citation.get("chunk_id", "")

    header = f"[{ref}] {ticker} | {filing_type} | {period}"

    with st.expander(header, expanded=False):
        st.markdown(
            f"""
            <div style="
                background: #1e293b;
                border-left: 3px solid #38bdf8;
                padding: 12px 16px;
                border-radius: 6px;
                color: #e2e8f0;
                font-size: 0.88rem;
                line-height: 1.6;
                margin-bottom: 8px;
            ">
                {excerpt}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"chunk_id: `{chunk_id}`")
