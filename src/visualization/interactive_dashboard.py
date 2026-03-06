import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


def _iteration_sort_key(name: str):
    if name.startswith("iter_"):
        try:
            return (0, int(name.split("_")[1]))
        except (IndexError, ValueError):
            return (0, 999)
    if name == "final":
        return (1, 999)
    return (2, 999)


def _parse_stop_state(stopping_reason: str) -> str:
    if not stopping_reason:
        return "unknown"
    return stopping_reason.split(":", 1)[0].strip().lower()


def load_iteration_summaries(iter_dir: Path):
    texts = {}
    for fname in sorted(iter_dir.glob("*.txt")):
        texts[fname.stem] = fname.read_text(encoding="utf-8").strip()
    return texts


@st.cache_data(show_spinner=False)
def _get_embedding_cached(text: str) -> list[float]:
    if not text.strip():
        return [0.0] * 1536

    from openai import OpenAI

    client = OpenAI()
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return emb.data[0].embedding


def compute_semantic_drift(iter_texts):
    names = sorted(iter_texts.keys(), key=_iteration_sort_key)
    embeddings = [np.array(_get_embedding_cached(iter_texts[n])) for n in names]

    if not embeddings:
        return [], []

    s0_emb = embeddings[0]
    sims = [cosine_similarity([s0_emb], [emb])[0][0] for emb in embeddings]
    drift = [1 - sim for sim in sims]
    return names, drift


def load_cohort_records(example_root: Path) -> list[dict]:
    records: list[dict] = []
    if not example_root.exists():
        return records

    for lecture_dir in sorted([p for p in example_root.glob("lecture*") if p.is_dir()]):
        result_path = lecture_dir / "result.json"
        pairwise_path = lecture_dir / "pairwise_s0_vs_refined.json"

        if not result_path.exists():
            continue

        result = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = result.get("refinement_metadata", {})
        stop_reason = metadata.get("stopping_reason", "")

        pairwise_win_refined = None
        pairwise_win_s0 = None
        if pairwise_path.exists():
            pairwise = json.loads(pairwise_path.read_text(encoding="utf-8"))
            pairwise_win_refined = pairwise.get("win_rate", {}).get("gpt5_refined")
            pairwise_win_s0 = pairwise.get("win_rate", {}).get("gpt5_S0")

        lever_history = metadata.get("lever_history", [])
        faithfulness_start = None
        faithfulness_end = None
        faithfulness_delta = None
        if lever_history:
            faithfulness_start = float(lever_history[0].get("faithfulness", 0.0))
            faithfulness_end = float(lever_history[-1].get("faithfulness", 0.0))
            faithfulness_delta = faithfulness_end - faithfulness_start

        records.append(
            {
                "lecture": lecture_dir.name,
                "final_score": float(result.get("final_score_0to1", 0.0)),
                "stop_state": _parse_stop_state(stop_reason),
                "iterations": int(metadata.get("iterations_completed", 0)),
                "pairwise_refined_win_rate": pairwise_win_refined,
                "pairwise_s0_win_rate": pairwise_win_s0,
                "faithfulness_delta": faithfulness_delta,
                "faithfulness_start": faithfulness_start,
                "faithfulness_end": faithfulness_end,
                "stopping_reason": stop_reason,
            }
        )

    return records


load_dotenv()
st.set_page_config(page_title="Evaluation Dashboard", layout="wide")

st.title("Interactive Evaluation Dashboard")

root_dir = Path("data/summaries/refined_iterations")
lectures = sorted([p.name for p in root_dir.glob("*") if p.is_dir()])

if not lectures:
    st.error("No lecture directories found in data/summaries/refined_iterations/")
    st.stop()

lecture_id = st.sidebar.selectbox("Select Lecture", lectures)
iter_dir = root_dir / lecture_id
result_json = iter_dir / "result.json"

if not result_json.exists():
    st.error(f"result.json not found for lecture '{lecture_id}'")
    st.stop()

result = json.loads(result_json.read_text(encoding="utf-8"))
iter_texts = load_iteration_summaries(iter_dir)
iter_lengths = {k: len(v.split()) for k, v in iter_texts.items()}
iter_steps = sorted(iter_lengths.keys(), key=_iteration_sort_key)

signals = result.get("signals", {})
rubric = result.get("rubric", {})
agreement = result.get("agreement", {}).get("agreement_1to5", 0.0)
comprehensive = result.get("comprehensive_scoring", {})
hybrid = result.get("hybrid_scoring", {})
metadata = result.get("refinement_metadata", {})

final_score = result.get("final_score_0to1", 0.0)
comp_score = comprehensive.get("final_score", 0.0)
manual_score = hybrid.get("manual_weighted_score", 0.0)
disagreement = hybrid.get("scorer_disagreement_delta", abs(comp_score - manual_score))
stop_reason = metadata.get("stopping_reason", "")
stop_state = _parse_stop_state(stop_reason)
iterations_completed = metadata.get("iterations_completed", 0)
quality_history = metadata.get("quality_history", [])
lever_history = metadata.get("lever_history", [])
detected_domain = comprehensive.get("detected_domain", rubric.get("detected_domain", "unknown"))

st.subheader(f"Lecture: {lecture_id}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Final Score (0-1)", f"{final_score:.3f}")
m2.metric("Stop State", stop_state)
m3.metric("Iterations", str(iterations_completed))
m4.metric("Detected Domain", str(detected_domain).upper())
m5.metric("Scorer Delta |C-M|", f"{disagreement:.3f}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Summary Length Across Iterations")
    fig = px.line(
        x=iter_steps,
        y=[iter_lengths[k] for k in iter_steps],
        markers=True,
        labels={"x": "Iteration", "y": "Word Count"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Quality Score Trajectory")
    if quality_history:
        qx = list(range(1, len(quality_history) + 1))
        fig = px.line(
            x=qx,
            y=quality_history,
            markers=True,
            labels={"x": "Iteration", "y": "Quality Score"},
            range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quality history found in refinement metadata.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Final Deterministic Signals")
    fig = px.bar(
        x=["Length Error", "Coverage", "Glossary Recall", "Hallucination"],
        y=[
            signals.get("length_error", 0.0),
            signals.get("section_coverage_pct", 0.0),
            signals.get("glossary_recall", 0.0),
            signals.get("suspected_hallucination_rate", 0.0),
        ],
        range_y=[0, 1],
        labels={"x": "Signal", "y": "Value"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Score Components")
    fig = px.bar(
        x=["Final", "Comprehensive", "Manual"],
        y=[final_score, comp_score, manual_score],
        range_y=[0, 1],
        labels={"x": "Score Type", "y": "Score"},
    )
    st.plotly_chart(fig, use_container_width=True)

col5, col6 = st.columns(2)

with col5:
    st.subheader("Final Rubric Radar")
    categories = ["Coverage", "Faithfulness", "Organization", "Clarity", "Style"]
    values = [
        float(rubric.get("coverage", 0)),
        float(rubric.get("faithfulness", 0)),
        float(rubric.get("organization", 0)),
        float(rubric.get("clarity", 0)),
        float(rubric.get("style", 0)),
    ]
    values += values[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color="royalblue"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col6:
    st.subheader("Rubric Average Trend")
    if lever_history:
        avg_trend = [sum(step.values()) / len(step) for step in lever_history]
        fig = px.line(
            x=list(range(1, len(avg_trend) + 1)),
            y=avg_trend,
            markers=True,
            labels={"x": "Iteration", "y": "Average Rubric (1-5)"},
            range_y=[0, 5],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No lever history found in refinement metadata.")

with st.expander("Run Diagnostics", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Stopping reason: {stop_reason or 'N/A'}")
        st.write(f"Agreement score (1-5): {agreement:.2f}")
        st.write(f"Comprehensive score: {comp_score:.3f}")
        st.write(f"Manual weighted score: {manual_score:.3f}")
    with c2:
        st.write(f"Final quality score: {metadata.get('final_quality_score', 0.0):.3f}")
        st.write(f"Hallucination damping factor: {hybrid.get('hallucination_damping_factor', 0.0):.3f}")
        st.write(f"Scorer disagreement delta: {disagreement:.3f}")
        st.write(f"Target words: {metadata.get('target_words', 'N/A')}")

with st.expander("Semantic Drift (optional, API call)", expanded=False):
    enable_drift = st.checkbox(
        "Compute semantic drift via embeddings",
        value=False,
        help="Uses OpenAI embeddings; slower and incurs API usage.",
    )
    if enable_drift:
        try:
            names, drifts = compute_semantic_drift(iter_texts)
            fig = px.line(
                x=names,
                y=drifts,
                markers=True,
                labels={"x": "Iteration", "y": "Semantic Drift"},
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to compute drift: {exc}")

st.subheader("Iteration Summaries")
for name in iter_steps:
    with st.expander(f"{name}.txt"):
        st.write(iter_texts[name])

st.markdown("---")
st.header("Cohort Insights (example_run)")

example_root = Path("example_run")
cohort_records = load_cohort_records(example_root)

if not cohort_records:
    st.info("No cohort records found under example_run/. Run evaluations and populate example_run first.")
else:
    cohort_df = pd.DataFrame(cohort_records).sort_values("lecture")

    st.subheader("Cross-Lecture Comparison")
    display_df = cohort_df[
        [
            "lecture",
            "final_score",
            "stop_state",
            "iterations",
            "pairwise_refined_win_rate",
        ]
    ].rename(
        columns={
            "lecture": "Lecture",
            "final_score": "Final Score (0-1)",
            "stop_state": "Stop State",
            "iterations": "Iterations",
            "pairwise_refined_win_rate": "Pairwise Refined Win Rate",
        }
    )
    st.dataframe(display_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            cohort_df,
            x="lecture",
            y="final_score",
            color="stop_state",
            title="Final Score by Lecture",
            labels={"lecture": "Lecture", "final_score": "Final Score (0-1)", "stop_state": "Stop State"},
            range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        stop_counts = cohort_df["stop_state"].value_counts().reset_index()
        stop_counts.columns = ["stop_state", "count"]
        fig = px.pie(
            stop_counts,
            names="stop_state",
            values="count",
            title="Convergence Panel: Stop-State Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ablation Mini-Chart")
    st.caption(
        "Fixed-iteration baseline artifacts are not present in example_run. "
        "Using available baseline proxy: S0 vs Lever/Hybrid pairwise win-rate comparison."
    )
    ablation_df = cohort_df[["lecture", "pairwise_s0_win_rate", "pairwise_refined_win_rate"]].copy()
    ablation_df = ablation_df.rename(
        columns={
            "pairwise_s0_win_rate": "S0 Baseline",
            "pairwise_refined_win_rate": "Lever/Hybrid",
        }
    )
    ablation_long = ablation_df.melt(id_vars=["lecture"], var_name="Method", value_name="Win Rate")
    fig = px.bar(
        ablation_long,
        x="lecture",
        y="Win Rate",
        color="Method",
        barmode="group",
        range_y=[0, 1],
        labels={"lecture": "Lecture"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Error-Analysis Callout")
    faithfulness_df = cohort_df.dropna(subset=["faithfulness_delta"]).sort_values("faithfulness_delta", ascending=False)
    if not faithfulness_df.empty:
        top = faithfulness_df.iloc[0]
        st.info(
            f"Concrete improvement example: {top['lecture']} improved faithfulness from "
            f"{top['faithfulness_start']:.2f} to {top['faithfulness_end']:.2f} "
            f"(Δ = {top['faithfulness_delta']:+.2f}) over {int(top['iterations'])} iterations."
        )
    else:
        st.info("Faithfulness trend data unavailable for cohort error-analysis callout.")

    st.subheader("Takeaways")
    avg_final = float(cohort_df["final_score"].mean()) if not cohort_df.empty else 0.0
    avg_iters = float(cohort_df["iterations"].mean()) if not cohort_df.empty else 0.0
    avg_refined_win = float(cohort_df["pairwise_refined_win_rate"].dropna().mean()) if not cohort_df["pairwise_refined_win_rate"].dropna().empty else 0.0
    pass_like = cohort_df["stop_state"].isin(["pass", "borderline"]).mean() if not cohort_df.empty else 0.0

    st.markdown(
        "\n".join(
            [
                f"- Across {len(cohort_df)} lectures, average final score is **{avg_final:.3f}**.",
                f"- Average refinement depth is **{avg_iters:.1f} iterations**, showing consistent convergence behavior.",
                f"- Lever/hybrid summaries win pairwise against S0 at **{avg_refined_win:.2%}** on average, with **{pass_like:.0%}** ending in pass/borderline states.",
            ]
        )
    )
