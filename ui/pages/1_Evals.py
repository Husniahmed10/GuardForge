# ─────────────────────────────────────────────────────────────────────────────
# Eval Suite page — auto-discovered by Streamlit multi-page routing.
# Accessible from the sidebar when running: streamlit run ui/app.py
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

# Add project root to path so evals.* imports work correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals")

import asyncio
import nest_asyncio
import pandas as pd
import streamlit as st

nest_asyncio.apply()

from evals.pipeline import run_pipeline, load_golden_dataset
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics

SCORE_COLORS = {"green": "#d4edda", "yellow": "#fff3cd", "red": "#f8d7da"}

def _badge(score):
    return "🟢" if score >= 0.75 else ("🟡" if score >= 0.5 else "🔴")

def _grade(score):
    return "✅ Good" if score >= 0.75 else ("⚠️ Fair" if score >= 0.5 else "❌ Poor")

def _color_score(val):
    if not isinstance(val, (int, float)): return ""
    return f"background-color: {SCORE_COLORS['green']}" if val >= 0.75 else (f"background-color: {SCORE_COLORS['yellow']}" if val >= 0.5 else f"background-color: {SCORE_COLORS['red']}")

def _render_metric_table(df, metric_col, title):
    avg = df[metric_col].mean()
    st.markdown(f"**{title}** — AVG: {_badge(avg)} `{avg:.2f}` {_grade(avg)}")
    styled = df.style.applymap(_color_score, subset=[metric_col]).format({metric_col: "{:.3f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

for key, default in [("golden", None), ("pipeline_done", False), ("enriched_dataset", None),
                      ("guardrails_results", None), ("metric_results", None), ("pipeline_rows", [])]:
    if key not in st.session_state:
        st.session_state[key] = load_golden_dataset() if key == "golden" else default

golden = st.session_state.golden
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("🧪 Enterprise RAG — Evaluation Suite")
st.caption("Step 1: Review ground truth → Step 2: Run live pipeline → Step 3: Score with RAGAS")
st.info(f"🔗 Backend: `{backend_url}`", icon="ℹ️")
st.divider()

tab1, tab2, tab3 = st.tabs(["📋 Step 1 — Ground Truth", "🚀 Step 2 — Live Pipeline", "📊 Step 3 — Eval Metrics"])

with tab1:
    st.subheader("Ground Truth Dataset")
    rag_rows = [{"ID": s["id"], "Domain": s["domain"].replace("_"," ").title(), "Question": s["question"],
                 "Reference Answer": s["reference"][:120]+"..." if len(s["reference"])>120 else s["reference"],
                 "Expected Tool": s["expected_tools"][0] if s["expected_tools"] else "—"} for s in golden["rag_samples"]]
    st.dataframe(pd.DataFrame(rag_rows), use_container_width=True, hide_index=True)
    st.caption(f"✅ {len(rag_rows)} golden RAG samples")
    st.divider()
    st.subheader("Guardrails Test Cases")
    g_rows = [{"ID": g["id"], "Input": g["input"], "Expected": "🛡️ Block" if g["expected_blocked"] else "✅ Pass",
               "Type": g["type"], "Description": g["description"]} for g in golden["guardrails_samples"]]
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)
    with st.expander("View raw golden_dataset.json"):
        st.json(golden)

with tab2:
    st.subheader("Live Pipeline — Collect Real Responses")
    st.markdown(f"Sends golden questions to **`{backend_url}/query`** and captures real responses.")

    col_p1, col_p2, _ = st.columns([1, 1, 2])
    run_pipeline_btn = col_p1.button("▶️ Run Live Pipeline", type="primary", use_container_width=True, disabled=st.session_state.pipeline_done)
    reset_btn = col_p2.button("🔄 Reset & Re-run", use_container_width=True, disabled=not st.session_state.pipeline_done)

    if reset_btn:
        for k in ["pipeline_done","enriched_dataset","guardrails_results","metric_results","pipeline_rows"]:
            st.session_state[k] = False if k == "pipeline_done" else ([] if k == "pipeline_rows" else None)
        st.rerun()

    if run_pipeline_btn:
        st.session_state.pipeline_rows = []
        pb = st.progress(0, text="Starting...")
        live_slot = st.empty()

        def pipeline_cb(i, total, question, stage, response=""):
            pct = int((i / total) * 100)
            if stage == "calling":
                pb.progress(pct, text=f"[{i+1}/{total}] Calling: {question[:60]}...")
            else:
                st.session_state.pipeline_rows.append({"#": i+1, "Question": question[:55], "Response": response[:80] if response else "⚠️", "Status": "✅" if response else "❌"})
                live_slot.dataframe(pd.DataFrame(st.session_state.pipeline_rows), use_container_width=True, hide_index=True)
                pb.progress(int(((i+1)/total)*100), text=f"[{i+1}/{total}] ✅ Done")

        enriched = run_pipeline(golden, progress_callback=pipeline_cb)
        st.session_state.enriched_dataset = enriched
        pb.progress(100, text="✅ Done!")

        st.divider()
        st.subheader("Guardrails Tests")
        gp = st.progress(0, text="Running guardrails tests...")
        g_results = run_guardrails_eval(enriched["guardrails_samples"], progress_callback=lambda i, t, inp: gp.progress(int((i/t)*100), text=f"[{i+1}/{t}] {inp[:60]}..."))
        g_metrics = compute_guardrails_metrics(g_results)
        st.session_state.guardrails_results = g_results
        st.session_state.pipeline_done = True
        gp.progress(100, text="✅ Complete!")

        LABELS = {"TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅", "FP": "🛡️ Blocked ❌ FP", "FN": "✅ Passed ❌ Missed"}
        st.dataframe(pd.DataFrame([{"ID": r["id"], "Input": r["input"][:70], "Expected": "🛡️ Block" if r["expected_blocked"] else "✅ Pass",
                                    "Actual": "Blocked" if r["actual_blocked"] else "Passed", "Result": LABELS.get(r["result"], r["result"])} for r in g_results]), use_container_width=True, hide_index=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Correct", f"{g_metrics['correct']}/{g_metrics['total']}")
        mc2.metric("Precision", f"{g_metrics['precision']:.2f}")
        mc3.metric("Recall", f"{g_metrics['recall']:.2f}")
        mc4.metric("Accuracy", f"{g_metrics['accuracy']:.2f}")

    elif st.session_state.pipeline_done:
        st.success("✅ Pipeline already run.")
        resp_rows = [{"#": s["id"], "Domain": s["domain"].replace("_"," ").title(), "Question": s["question"][:60],
                      "Live Response": s.get("actual_response","")[:100], "Tool": s.get("actual_tools_called",["—"])[0]} for s in st.session_state.enriched_dataset["rag_samples"]]
        st.dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Eval Metrics — RAGAS + Tool Correctness")
    if not st.session_state.pipeline_done:
        st.warning("⚠️ Complete Step 2 first.")
    else:
        run_metrics_btn = st.button("▶️ Run Eval Metrics", type="primary")
        METRIC_NAMES = {"faithfulness":"Exp 1 — Faithfulness","answer_relevancy":"Exp 2 — Answer Relevancy",
                        "context_precision":"Exp 3 — Context Precision","context_recall":"Exp 4 — Context Recall",
                        "answer_correctness":"Exp 5 — Answer Correctness","tool_correctness":"Exp 6 — Tool Correctness"}
        if run_metrics_btn:
            status_slot = st.empty()
            metric_results = _run_async(run_all_metrics(st.session_state.enriched_dataset, status_cb=lambda m: status_slot.info(m)))
            st.session_state.metric_results = metric_results
            status_slot.success("✅ All 6 experiments complete!")
            for key, title in METRIC_NAMES.items():
                if key in metric_results:
                    _render_metric_table(metric_results[key], key, title)
        elif st.session_state.metric_results:
            for key, title in METRIC_NAMES.items():
                if key in st.session_state.metric_results:
                    _render_metric_table(st.session_state.metric_results[key], key, title)

        if st.session_state.metric_results:
            st.divider()
            st.subheader("Final Summary")
            mr = st.session_state.metric_results
            summary = [(n, mr.get(k, pd.DataFrame()).get(k, pd.Series()).mean()) for k, n in METRIC_NAMES.items()]
            cols = st.columns(len(summary))
            for col, (name, score) in zip(cols, summary):
                if pd.notna(score): col.metric(name, f"{score:.2f}", _grade(score))
            st.dataframe(pd.DataFrame([{"Metric": n, "Score": f"{s:.3f}" if pd.notna(s) else "—", "Grade": _grade(s) if pd.notna(s) else "—"} for n, s in summary]), use_container_width=True, hide_index=True)
