"""FinSight – Streamlit application entrypoint."""
import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FinSight – Earnings Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        .sidebar-logo {
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .sidebar-tagline {
            font-size: 0.75rem;
            color: #94a3b8 !important;
            margin-top: -4px;
        }
        .status-badge {
            display: inline-block;
            background: #064e3b;
            color: #6ee7b7 !important;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📈 FinSight</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Earnings Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        options=["💬 Chat", "📊 Eval Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Pipeline Status")

    try:
        resp = httpx.get(f"{BACKEND_URL}/documents/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("total_count", 0)
            st.markdown(
                f'<span class="status-badge">✅ {total} filings indexed</span>',
                unsafe_allow_html=True,
            )
            filings = data.get("filings", [])
            tickers = sorted({f["ticker"] for f in filings})
            if tickers:
                st.caption("Tickers: " + ", ".join(tickers[:10]) + ("…" if len(tickers) > 10 else ""))
        else:
            st.warning("Backend returned an error.")
    except Exception:
        st.error("⚠️ Backend unreachable")

    st.markdown("---")
    st.caption(f"Backend: `{BACKEND_URL}`")

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "💬 Chat":
    from frontend.pages.chat import render_chat_page
    render_chat_page(BACKEND_URL)
else:
    from frontend.pages.eval_dashboard import render_eval_dashboard
    render_eval_dashboard(BACKEND_URL)
