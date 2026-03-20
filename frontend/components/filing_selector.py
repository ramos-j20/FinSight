"""Filing selector component for filtering queries by ticker and filing type."""
from typing import Literal

import httpx
import streamlit as st


def render_filing_selector(
    backend_url: str,
) -> tuple[str | None, Literal["10-K", "10-Q"] | None]:
    """Render ticker + filing-type filter widgets.

    Returns:
        (ticker_filter, filing_type_filter): Both may be None for "all".
    """
    col1, col2 = st.columns([2, 2])

    with col1:
        MAG_7_OPTIONS = {
            "🍎 Apple (AAPL)": "AAPL",
            "🪟 Microsoft (MSFT)": "MSFT",
            "🔍 Alphabet (GOOGL)": "GOOGL",
            "📦 Amazon (AMZN)": "AMZN",
            "♾️ Meta (META)": "META",
            "🖥️ Nvidia (NVDA)": "NVDA",
            "🚗 Tesla (TSLA)": "TSLA"
        }
        
        selected_displays = st.multiselect(
            "🏢 Ticker Selection",
            options=list(MAG_7_OPTIONS.keys()),
            default=[],
            placeholder="Select companies (leave empty for all)",
            key="filing_selector_ticker",
        )
        
        if not selected_displays:
            ticker = None
        else:
            ticker = ",".join([MAG_7_OPTIONS[d] for d in selected_displays])

    with col2:
        filing_type_option = st.selectbox(
            "📄 Filing Type",
            options=["All", "10-K", "10-Q"],
            key="filing_selector_type",
        )
        filing_type: Literal["10-K", "10-Q"] | None = None if filing_type_option == "All" else filing_type_option  # type: ignore[assignment]

    # Validate ticker against the index
    if ticker:
        try:
            resp = httpx.get(
                f"{backend_url}/documents/",
                params={"ticker": ticker, "page_size": 1},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("total_count", 0) == 0:
                    st.warning(
                        f"⚠️ Ticker **{ticker}** not found in the index. "
                        "Results may be empty. Ingest this ticker first."
                    )
                else:
                    count = data["total_count"]
                    st.success(f"✅ {count} filing(s) indexed for **{ticker}**.")
        except Exception:
            st.warning("Could not validate ticker — backend may be unavailable.")

    return ticker, filing_type
