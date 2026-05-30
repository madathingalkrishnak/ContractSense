"""
src/dashboard/app.py
---------------------
ContractSense Streamlit Dashboard

Pages:
  1. Overview       — Eval run history, overall pass rates, trend charts
  2. Trace Inspector — Drill into individual QA pairs, see judge scores + explanations
  3. Failure Analysis — Browse failed evaluations, find patterns
  4. Live QA         — Ask questions against the live RAG system
  5. Run Comparison  — Compare two eval runs side by side

Run:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from configs.config_loader import load_config

cfg = load_config()

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ContractSense",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .metric-card {
        background: #0f1117;
        border: 1px solid #2d2d2d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #00ff88;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .pass-badge {
        background: #00ff8820;
        color: #00ff88;
        border: 1px solid #00ff8840;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
    }
    .fail-badge {
        background: #ff444420;
        color: #ff4444;
        border: 1px solid #ff444440;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
    }
    .stMetric {
        background: #0f1117;
        border: 1px solid #2d2d2d;
        border-radius: 8px;
        padding: 16px;
    }
    code {
        font-family: 'IBM Plex Mono', monospace;
        background: #1e1e2e;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .sidebar-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: #00ff88;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d2d2d;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading helpers ──────────────────────────────────────────


@st.cache_data(ttl=30)   # refresh every 30 seconds
def load_all_runs() -> list[dict]:
    """Load summaries from all eval runs."""
    runs_dir = cfg.abs_path("evals_runs")
    summaries = []

    for run_dir in sorted(runs_dir.glob("eval_*"), reverse=True):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                summaries.append(summary)
            except Exception:
                continue

    return summaries


@st.cache_data(ttl=30)
def load_run_results(run_id: str) -> pd.DataFrame:
    """Load detailed results for a specific eval run."""
    results_path = cfg.abs_path("evals_runs") / run_id / "results.csv"
    if results_path.exists():
        return pd.read_csv(results_path)
    return pd.DataFrame()


def score_color(score: float) -> str:
    """Return color string based on score."""
    if score >= 0.8:
        return "#00ff88"
    elif score >= 0.6:
        return "#ffcc00"
    else:
        return "#ff4444"

def badge_text(passed: bool) -> str:
    return "✓" if passed else "✗"

# ── Sidebar navigation ────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-header">⚖️ ContractSense</div>', unsafe_allow_html=True)
    st.caption("LLM Evaluation Platform")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Trace Inspector", "❌ Failure Analysis", "💬 Live QA", "📈 Run Comparison"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("**Config**")
    st.code(f"Model: {cfg.llm.model}", language=None)
    st.code(f"Embeddings: {cfg.embedding.model.split('/')[-1]}", language=None)
    st.code(f"Top-K: {cfg.rag.top_k}", language=None)
    st.code(f"Chunk: {cfg.ingestion.chunk_size}c / {cfg.ingestion.chunk_overlap}ov", language=None)


# ── Page: Overview ────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Evaluation Overview")

    runs = load_all_runs()

    if not runs:
        st.warning(
            "No evaluation runs found yet.\n\n"
            "Run the pipeline first:\n"
            "```bash\npython scripts/run_pipeline.py --dev\n```"
        )
        st.stop()

    # Latest run summary
    latest = runs[0]
    st.subheader(f"Latest Run: `{latest.get('run_name', latest.get('run_id', 'unknown'))}`")
    st.caption(f"Timestamp: {latest.get('timestamp', 'unknown')}")

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        val = latest.get("overall_pass_rate", 0)
        st.metric("Overall Pass Rate", f"{val:.1%}", delta=None)

    with col2:
        val = latest.get("mean_faithfulness", 0)
        st.metric("Faithfulness", f"{val:.3f}")

    with col3:
        val = latest.get("mean_hallucination_score", 0)
        st.metric("Hallucination Score", f"{val:.3f}")

    with col4:
        val = latest.get("mean_answer_relevancy", 0)
        st.metric("Answer Relevancy", f"{val:.3f}")

    st.divider()

    # Trend chart: pass rate over time
    if len(runs) > 1:
        st.subheader("📈 Pass Rate Trend")
        df_trend = pd.DataFrame([
            {
                "run": r.get("run_name", r.get("run_id", "?"))[-12:],
                "timestamp": r.get("timestamp", ""),
                "overall_pass_rate": r.get("overall_pass_rate", 0),
                "mean_faithfulness": r.get("mean_faithfulness", 0),
                "mean_hallucination_score": r.get("mean_hallucination_score", 0),
                "mean_answer_relevancy": r.get("mean_answer_relevancy", 0),
            }
            for r in reversed(runs)
        ])

        fig = go.Figure()
        metrics_to_plot = [
            ("overall_pass_rate", "#00ff88", "Pass Rate"),
            ("mean_faithfulness", "#4488ff", "Faithfulness"),
            ("mean_hallucination_score", "#ff8844", "Hallucination"),
            ("mean_answer_relevancy", "#ff44aa", "Relevancy"),
        ]
        for col_name, color, label in metrics_to_plot:
            fig.add_trace(go.Scatter(
                x=df_trend["run"],
                y=df_trend[col_name],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))

        fig.update_layout(
            template="plotly_dark",
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,17,23,1)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # All runs table
    st.subheader("All Evaluation Runs")
    df_runs = pd.DataFrame([
        {
            "Run ID": r.get("run_id", "?"),
            "Name": r.get("run_name", "?"),
            "Timestamp": r.get("timestamp", "?"),
            "Rows": r.get("num_evaluated", "?"),
            "Pass Rate": f"{r.get('overall_pass_rate', 0):.1%}",
            "Faithfulness": f"{r.get('mean_faithfulness', 0):.3f}",
            "Hallucination": f"{r.get('mean_hallucination_score', 0):.3f}",
            "Relevancy": f"{r.get('mean_answer_relevancy', 0):.3f}",
        }
        for r in runs
    ])
    st.dataframe(df_runs, use_container_width=True, hide_index=True)


# ── Page: Trace Inspector ─────────────────────────────────────────

elif page == "🔍 Trace Inspector":
    st.title("🔍 Trace Inspector")
    st.caption("Drill into individual question-answer pairs and their judge scores")

    runs = load_all_runs()
    if not runs:
        st.warning("No eval runs found. Run the pipeline first.")
        st.stop()

    run_options = {
        r.get("run_name", r.get("run_id", "?")): r.get("run_id", "?")
        for r in runs
    }
    selected_run_name = st.selectbox("Select Run", list(run_options.keys()))
    selected_run_id = run_options[selected_run_name]

    df = load_run_results(selected_run_id)

    if df.empty:
        st.warning(f"No results found for run: {selected_run_id}")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_passed = st.selectbox("Filter", ["All", "Passed Only", "Failed Only"])
    with col2:
        persona_options = ["All"] + sorted(df["persona"].dropna().unique().tolist()) if "persona" in df.columns else ["All"]
        filter_persona = st.selectbox("Persona", persona_options)
    with col3:
        sort_by = st.selectbox("Sort by", ["avg_score ↑", "avg_score ↓", "question"])

    # Apply filters
    filtered_df = df.copy()
    if filter_passed == "Passed Only":
        filtered_df = filtered_df[filtered_df["passed_all"] == True]
    elif filter_passed == "Failed Only":
        filtered_df = filtered_df[filtered_df["passed_all"] == False]

    if filter_persona != "All" and "persona" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["persona"] == filter_persona]

    if "avg_score" in filtered_df.columns:
        ascending = "↑" in sort_by
        filtered_df = filtered_df.sort_values("avg_score", ascending=ascending)

    st.caption(f"Showing {len(filtered_df)} of {len(df)} rows")

    # Row browser
    for idx, row in filtered_df.head(20).iterrows():
        passed = row.get("passed_all", False)
        avg = row.get("avg_score", 0)
        badge = f'<span class="pass-badge">PASS</span>' if passed else f'<span class="fail-badge">FAIL</span>'

        with st.expander(
            f"{badge_text(passed)} [{avg:.2f}] {str(row.get('question', ''))[:80]}",
            expanded=False,
        ):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Question**")
                st.info(row.get("question", ""))
                st.markdown("**Answer**")
                st.success(str(row.get("answer_preview", ""))[:500])

            with col2:
                st.markdown("**Scores**")
                score_cols = [c for c in df.columns if c.startswith("score_")]
                for sc in score_cols:
                    metric = sc[6:]  # strip "score_"
                    score_val = row.get(sc, 0)
                    exp_col = f"explanation_{metric}"
                    explanation = row.get(exp_col, "")
                    color = score_color(score_val)
                    st.markdown(
                        f"**{metric}**: "
                        f"<span style='color:{color};font-family:monospace'>"
                        f"{score_val:.3f}</span>",
                        unsafe_allow_html=True,
                    )
                    if explanation:
                        st.caption(f"↳ {explanation}")

            if "persona" in row:
                st.caption(f"Persona: {row['persona']}  |  Contract: {row.get('contract', 'unknown')}")


# ── Page: Failure Analysis ────────────────────────────────────────

elif page == "❌ Failure Analysis":
    st.title("❌ Failure Analysis")
    st.caption("Understand why evaluations fail — find systematic patterns")

    runs = load_all_runs()
    if not runs:
        st.warning("No eval runs found.")
        st.stop()

    run_options = {r.get("run_name", r.get("run_id", "?")): r.get("run_id", "?") for r in runs}
    selected_run_id = run_options[st.selectbox("Select Run", list(run_options.keys()))]

    df = load_run_results(selected_run_id)
    if df.empty:
        st.stop()

    failures = df[df["passed_all"] == False].copy() if "passed_all" in df.columns else df

    st.metric("Failure Count", len(failures), delta=f"{len(failures)/len(df):.0%} of total")

    if failures.empty:
        st.success("No failures in this run!")
        st.stop()

    # Failure breakdown by metric
    st.subheader("Failures by Metric")
    passed_cols = [c for c in df.columns if c.startswith("passed_")]
    if passed_cols:
        fail_counts = {col[7:]: (~df[col].astype(bool)).sum() for col in passed_cols}
        fig = px.bar(
            x=list(fail_counts.keys()),
            y=list(fail_counts.values()),
            labels={"x": "Metric", "y": "Failure Count"},
            color_discrete_sequence=["#ff4444"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,17,23,1)",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    st.subheader("Score Distribution (Failures)")
    score_cols = [c for c in df.columns if c.startswith("score_")]
    if score_cols:
        fig = go.Figure()
        colors = ["#ff4444", "#ff8844", "#ffcc00", "#4488ff"]
        for i, col in enumerate(score_cols):
            metric = col[6:]
            fail_scores = failures[col].dropna()
            fig.add_trace(go.Histogram(
                x=fail_scores,
                name=metric,
                nbinsx=20,
                marker_color=colors[i % len(colors)],
                opacity=0.75,
            ))
        fig.update_layout(
            barmode="overlay",
            template="plotly_dark",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,17,23,1)",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Score",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Failure table
    st.subheader("Failure Cases")
    display_cols = ["question", "answer_preview", "avg_score"] + score_cols
    display_cols = [c for c in display_cols if c in failures.columns]
    st.dataframe(
        failures[display_cols].head(50).reset_index(drop=True),
        use_container_width=True,
    )


# ── Page: Live QA ─────────────────────────────────────────────────

elif page == "💬 Live QA":
    st.title("💬 Live Contract QA")
    st.caption("Ask questions against the live RAG system")

    # Check Ollama
    try:
        from src.rag.llm_client import check_ollama_running, check_model_available
        ollama_ok = check_ollama_running()
        model_ok = check_model_available() if ollama_ok else False
    except Exception:
        ollama_ok = model_ok = False

    col1, col2 = st.columns(2)
    with col1:
        status = "🟢 Running" if ollama_ok else "🔴 Not running"
        st.metric("Ollama Server", status)
    with col2:
        mstatus = f"🟢 {cfg.llm.model}" if model_ok else f"🔴 {cfg.llm.model} not found"
        st.metric("Model", mstatus)

    if not ollama_ok:
        st.error("Ollama is not running. Start it with: `ollama serve`")
        st.stop()

    if not model_ok:
        st.warning(f"Model not found. Pull it with: `ollama pull {cfg.llm.model}`")

    # Check vector store has data
    try:
        from src.ingestion.vector_store import get_vector_store, get_collection_stats
        store = get_vector_store()
        stats = get_collection_stats(store)
        n_chunks = stats["total_chunks"]
    except Exception as e:
        st.error(f"Could not connect to vector store: {e}")
        st.stop()

    if n_chunks == 0:
        st.warning("Vector store is empty. Run ingestion first.")
        st.stop()

    st.caption(f"📚 Knowledge base: {n_chunks:,} indexed chunks")

    question = st.text_input(
        "Ask a contract question",
        placeholder="What are the termination conditions?",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    if ask_button and question:
        with st.spinner("Retrieving and generating..."):
            try:
                from src.rag.chain import ContractRAG

                @st.cache_resource
                def get_rag():
                    return ContractRAG()

                rag = get_rag()
                response = rag.ask(question)

                st.markdown("### Answer")
                st.success(response.answer)

                st.markdown(f"*Latency: {response.latency_ms:.0f}ms  |  Sources: {', '.join(response.source_files)}*")

                with st.expander("📄 Retrieved Context", expanded=False):
                    for i, (chunk, score) in enumerate(
                        zip(response.retrieved_chunks, response.retrieval_scores)
                    ):
                        st.markdown(f"**Chunk {i+1}** — Score: `{score:.4f}` — `{chunk.metadata.get('file_name', 'unknown')}`")
                        st.text(chunk.page_content[:600])
                        st.divider()

            except Exception as e:
                st.error(f"Error: {e}")


# ── Page: Run Comparison ──────────────────────────────────────────

elif page == "📈 Run Comparison":
    st.title("📈 Run Comparison")
    st.caption("Compare two evaluation runs side by side")

    runs = load_all_runs()
    if len(runs) < 2:
        st.warning("Need at least 2 eval runs for comparison. Run the pipeline multiple times.")
        st.stop()

    run_options = {r.get("run_name", r.get("run_id", "?")): r for r in runs}
    run_names = list(run_options.keys())

    col1, col2 = st.columns(2)
    with col1:
        run_a_name = st.selectbox("Run A (baseline)", run_names, index=1)
    with col2:
        run_b_name = st.selectbox("Run B (new)", run_names, index=0)

    run_a = run_options[run_a_name]
    run_b = run_options[run_b_name]

    st.subheader("Metric Comparison")

    metrics = ["overall_pass_rate", "mean_faithfulness", "mean_hallucination_score", "mean_answer_relevancy"]
    metric_labels = ["Pass Rate", "Faithfulness", "Hallucination Score", "Relevancy"]

    comparison_data = []
    for m, label in zip(metrics, metric_labels):
        a_val = run_a.get(m, 0)
        b_val = run_b.get(m, 0)
        delta = b_val - a_val
        comparison_data.append({
            "Metric": label,
            "Run A": f"{a_val:.3f}",
            "Run B": f"{b_val:.3f}",
            "Delta": f"{delta:+.3f}",
            "Direction": "↑ Better" if delta > 0.01 else ("↓ Worse" if delta < -0.01 else "→ Same"),
        })

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # Radar chart
    fig = go.Figure()
    for run_data, name, color in [
        (run_a, run_a_name, "#4488ff"),
        (run_b, run_b_name, "#00ff88"),
    ]:
        values = [run_data.get(m, 0) for m in metrics]
        values.append(values[0])  # close the radar
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            name=name,
            line_color=color,
            fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color else color + "26",
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
