"""Evaluation dashboard page — historical metrics and charts."""
import httpx
import streamlit as st


def render_eval_dashboard(backend_url: str) -> None:
    """Render the retrieval evaluation dashboard."""
    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #f59e0b, #ef4444);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem; font-weight: 800; margin-bottom: 0;
        ">📊 Eval Dashboard</h1>
        <p style="color: #94a3b8; margin-top: 4px;">
            Retrieval quality metrics across evaluation runs.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Pipeline health ───────────────────────────────────────────────────────
    try:
        health_resp = httpx.get(f"{backend_url}/health", timeout=5)
        if health_resp.status_code == 200:
            hdata = health_resp.json()
            st.success(
                f"✅ Backend healthy — version `{hdata.get('version', 'unknown')}`"
            )
        else:
            st.warning("⚠️ Backend health check returned an unexpected status.")
    except Exception:
        st.error("❌ Backend unreachable. Start the API before using this dashboard.")

    st.markdown("---")

    # ── Run evaluation ────────────────────────────────────────────────────────
    st.subheader("🚀 Run Evaluation")
    eval_col1, eval_col2 = st.columns([3, 1])
    with eval_col1:
        dataset_path = st.text_input(
            "Eval dataset path (JSONL)",
            value="tests/fixtures/eval_dataset.jsonl",
            key="eval_dataset_path",
        )
    with eval_col2:
        run_eval = st.button("▶ Run Eval", use_container_width=True)

    if run_eval:
        with st.spinner("Running evaluation harness…"):
            try:
                resp = httpx.post(
                    f"{backend_url}/eval/run",
                    json={"eval_dataset_path": dataset_path},
                    timeout=300,
                )
                if resp.status_code == 200:
                    metrics = resp.json()
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Hit Rate", f"{metrics.get('hit_rate', 0):.3f}")
                    m_col2.metric("MRR", f"{metrics.get('mrr', 0):.3f}")
                    m_col3.metric("Avg Score", f"{metrics.get('avg_score', 0):.3f}")
                    m_col4.metric("Total Queries", metrics.get("total_queries", 0))
                    st.success("Evaluation complete! Results saved to database.")
                else:
                    st.error(f"Eval failed: {resp.text}")
            except Exception as exc:
                st.error(f"Error running eval: {exc}")

    st.markdown("---")

    # ── Model Routing Distribution ────────────────────────────────────────────
    st.subheader("🔀 Model Routing Distribution")
    try:
        routing_resp = httpx.get(f"{backend_url}/eval/routing-stats", timeout=10)
        if routing_resp.status_code == 200:
            routing_data = routing_resp.json()
            if routing_data:
                import pandas as pd
                routing_df = pd.DataFrame(routing_data)
                
                chart_df = routing_df.groupby("mode_used")["count"].sum().reset_index()
                st.bar_chart(chart_df.set_index("mode_used"), horizontal=True)
                
                st.markdown("**Average Latency per Mode**")
                modes = routing_df["mode_used"].unique()
                cols = st.columns(max(1, len(modes)))
                for i, m in enumerate(sorted(modes)):
                    avg_lat = routing_df[routing_df["mode_used"] == m]["avg_latency"].mean()
                    cols[i].metric(f"{m.capitalize()} Mode", f"{avg_lat:.0f} ms")
            else:
                st.info("No query logs available yet to show routing distribution.")
        else:
            st.warning("Could not fetch routing stats.")
    except Exception as exc:
        st.error(f"Failed to load routing stats: {exc}")

    st.markdown("---")

    # ── Historical results ────────────────────────────────────────────────────
    st.subheader("📜 Historical Eval Results")

    try:
        results_resp = httpx.get(f"{backend_url}/eval/results", timeout=10)
        if results_resp.status_code != 200:
            st.warning("Could not fetch eval results.")
            return

        rows = results_resp.json()

        if not rows:
            st.info(
                "No evaluation results yet. Run an evaluation above to populate this table."
            )
            return

        import pandas as pd

        df = pd.DataFrame(rows)

        # Rename / select display columns (gracefully handle missing cols)
        display_cols = [c for c in ["eval_run_id", "created_at", "hit_rate", "mrr", "faithfulness_score", "query", "expected_chunk_ids", "retrieved_chunk_ids"] if c in df.columns]
        df_display = df[display_cols].copy()

        # Make sure lists are converted to strings so Streamlit can render them safely
        if "expected_chunk_ids" in df_display.columns:
            df_display["expected_chunk_ids"] = df_display["expected_chunk_ids"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        
        if "retrieved_chunk_ids" in df_display.columns:
            df_display["retrieved_chunk_ids"] = df_display["retrieved_chunk_ids"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        col_renames = {
            "eval_run_id": "Run ID",
            "created_at": "Timestamp",
            "hit_rate": "Hit Rate",
            "mrr": "MRR",
            "faithfulness_score": "Avg Score",
            "query": "Query",
            "expected_chunk_ids": "Expected Chunks",
            "retrieved_chunk_ids": "Retrieved Chunks",
        }
        df_display = df_display.rename(columns={k: v for k, v in col_renames.items() if k in df_display.columns})

        # Highlight most recent row using index 0 (results are DESC ordered)
        def highlight_first(s: "pd.Series"):  # noqa: F821
            return ["background-color: #1e3a5f" if i == 0 else "" for i in range(len(s))]

        st.dataframe(
            df_display.style.apply(highlight_first, axis=0),
            use_container_width=True,
        )

        # ── Charts ────────────────────────────────────────────────────────────
        chart_df_cols = [c for c in ["created_at", "hit_rate", "mrr"] if c in df.columns]
        if len(chart_df_cols) == 3:
            chart_df = df[chart_df_cols].copy()
            chart_df = chart_df.rename(columns={"created_at": "timestamp", "hit_rate": "Hit Rate", "mrr": "MRR"})
            chart_df = chart_df.sort_values("timestamp")
            chart_df = chart_df.set_index("timestamp")

            st.markdown("---")
            st.subheader("📈 Retrieval Metrics Over Time")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Hit Rate**")
                st.line_chart(chart_df[["Hit Rate"]], color="#38bdf8")
            with c2:
                st.markdown("**MRR (Mean Reciprocal Rank)**")
                st.line_chart(chart_df[["MRR"]], color="#818cf8")

    except Exception as exc:
        st.error(f"Failed to load eval results: {exc}")
