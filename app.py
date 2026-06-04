from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import tensorflow as tf
except Exception:  # Streamlit should still render setup guidance if TF is unavailable.
    tf = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

from base_features import add_features, read_xlsb
from model_utils import score_dataframe_core

APP_TITLE = "Allocation AI — Keras FLM Ranker"
REQUIRED = ["allocation_model.keras", "deallocation_model.keras", "preprocessing_pipeline.joblib"]
METRIC_FILES = {
    "Allocation epoch log": "allocation_model_keras_epoch_log.csv",
    "Deallocation epoch log": "deallocation_model_keras_epoch_log.csv",
    "Allocation live metrics": "allocation_model_live_epoch_metrics.csv",
    "Deallocation live metrics": "deallocation_model_live_epoch_metrics.csv",
    "Cutoff sweep": "allocation_cutoff_sweep.csv",
    "File-level evaluation": "file_level_evaluation.csv",
    "Validation scored preview": "validation_scored_preview.csv",
}

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [p for p in REQUIRED if not Path(p).exists()]
    if tf is None:
        missing.append("tensorflow / keras runtime — install tensorflow-cpu>=2.21,<2.22 for Python 3.13, or redeploy with Python 3.11 for older TensorFlow")
    if missing:
        return None, None, None, missing
    allocation_model = tf.keras.models.load_model("allocation_model.keras")
    deallocation_model = tf.keras.models.load_model("deallocation_model.keras")
    bundle = joblib.load("preprocessing_pipeline.joblib")
    return allocation_model, deallocation_model, bundle, []


@st.cache_data(show_spinner=False)
def load_csv_if_exists(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def read_upload(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".xlsb"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsb") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            return read_xlsb(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    if name.endswith((".xlsx", ".xls")):
        # Most allocation workbooks use row 2 as the header on this sheet.
        try:
            return pd.read_excel(uploaded, sheet_name="3.3 Working Table", header=1)
        except Exception:
            return pd.read_excel(uploaded, header=0)
    return pd.read_csv(uploaded)


def model_input_dim(model) -> int | str:
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        return int(shape[-1])
    except Exception:
        return "Unknown"


def model_layer_table(model) -> pd.DataFrame:
    rows = []
    for i, layer in enumerate(getattr(model, "layers", []), start=1):
        cfg = layer.get_config() if hasattr(layer, "get_config") else {}
        rows.append({
            "#": i,
            "layer": layer.__class__.__name__,
            "name": getattr(layer, "name", ""),
            "units": cfg.get("units", ""),
            "activation": cfg.get("activation", ""),
            "dropout/rate": cfg.get("rate", ""),
            "parameters": int(layer.count_params()) if hasattr(layer, "count_params") else "",
        })
    return pd.DataFrame(rows)


def render_metric_cards(result: pd.DataFrame):
    final_sum = int(_num(result["AI Final Alloc"]).sum()) if "AI Final Alloc" in result else 0
    before_sum = int(_num(result["AI Allocation Before Deallocation"]).sum()) if "AI Allocation Before Deallocation" in result else 0
    dealloc_sum = int(_num(result["AI Deallocation Units"]).sum()) if "AI Deallocation Units" in result else 0
    safety_sum = int(_num(result["AI DC Safety Cut"]).sum()) if "AI DC Safety Cut" in result else 0
    neg_rows = int((_num(result["AI Left DC"]) < 0).sum()) if "AI Left DC" in result else 0
    allocated_rows = int((_num(result["AI Final Alloc"]) > 0).sum()) if "AI Final Alloc" in result else 0
    cols = st.columns(6)
    cols[0].metric("AI Final Alloc", f"{final_sum:,}")
    cols[1].metric("Before Dealloc", f"{before_sum:,}")
    cols[2].metric("Deallocated", f"{dealloc_sum:,}")
    cols[3].metric("DC Safety Cut", f"{safety_sum:,}")
    cols[4].metric("Rows Allocated", f"{allocated_rows:,}")
    cols[5].metric("Negative DC Rows", f"{neg_rows:,}")


def summarize_scored(result: pd.DataFrame) -> pd.DataFrame:
    final = _num(result.get("AI Final Alloc", pd.Series(0, index=result.index)))
    before = _num(result.get("AI Allocation Before Deallocation", pd.Series(0, index=result.index)))
    dealloc = _num(result.get("AI Deallocation Units", pd.Series(0, index=result.index)))
    left = _num(result.get("AI Left DC", pd.Series(0, index=result.index)))
    rows = [
        ("Total rows", len(result)),
        ("Rows with AI allocation", int((final > 0).sum())),
        ("Rows deallocated", int((dealloc > 0).sum())),
        ("Rows changed by deallocation/safety", int((before != final).sum())),
        ("Final allocation units", int(final.sum())),
        ("Starting allocation before deallocation", int(before.sum())),
        ("Total deallocated units", int(dealloc.sum())),
        ("Minimum AI Left DC", int(left.min() if len(left) else 0)),
        ("Negative AI Left DC rows", int((left < 0).sum())),
    ]
    if "Alloc. Rec." in result.columns:
        rec = _num(result["Alloc. Rec."])
        rows.extend([
            ("Alloc. Rec. units", int(rec.sum())),
            ("AI minus Alloc. Rec.", int(final.sum() - rec.sum())),
            ("Rows where AI > Alloc. Rec.", int((final > rec).sum())),
            ("Rows where AI < Alloc. Rec.", int((final < rec).sum())),
        ])
    if "Final Alloc." in result.columns and _num(result["Final Alloc."]).sum() > 0:
        actual = _num(result["Final Alloc."])
        mask = actual.notna()
        err = final - actual
        rows.extend([
            ("Actual Final Alloc units", int(actual.sum())),
            ("AI MAE vs existing Final Alloc", round(float(np.abs(err[mask]).mean()), 3)),
            ("AI bias vs existing Final Alloc", round(float(err[mask].mean()), 3)),
            ("Exact matches vs existing Final Alloc", int((final[mask] == actual[mask]).sum())),
        ])
    return pd.DataFrame(rows, columns=["Insight", "Value"])


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Scored Allocation")
    return bio.getvalue()


def plot_line(df: pd.DataFrame, cols: list[str], title: str):
    if not cols:
        return
    chart_df = df[["epoch"] + cols].dropna(how="all") if "epoch" in df.columns else df[cols]
    if px is not None and "epoch" in chart_df.columns:
        fig = px.line(chart_df, x="epoch", y=cols, title=title, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(chart_df.set_index("epoch") if "epoch" in chart_df.columns else chart_df)


def feature_family(name: str) -> str:
    n = str(name)
    if "cand" in n:
        return "FLM candidate features"
    if any(x in n for x in ["gap", "demand", "woc", "weekly", "l30", "d30", "d60", "ttm", "lw"]):
        return "Demand / velocity"
    if any(x in n for x in ["left", "dc", "supply", "qoh", "intrans", "transfer", "store_po"]):
        return "Supply / DC"
    if any(x in n for x in ["flm", "mil", "units", "rule", "rec"]):
        return "FLM / recommendation"
    if any(x in n for x in ["cost", "retail", "margin", "gm"]):
        return "Financial"
    if any(x in n for x in ["item", "site", "vendor", "brand", "class", "line", "dept"]):
        return "Group context"
    if "text" in n:
        return "Product text"
    return "Other"


st.title(APP_TITLE)
st.caption("Two separate Keras neural networks: one ranks individual FLMs to allocate, and one ranks FLMs to remove when DC is exceeded.")

allocation_model, deallocation_model, bundle, missing = load_artifacts()
if missing:
    st.error("Required runtime artifacts are missing or TensorFlow is not installed.")
    st.write("Add/install the following and restart Streamlit:")
    st.code("\n".join(missing))
    st.stop()

metrics = _read_json("model_metrics.json", {}) or {}
alloc_last = _read_json("allocation_model_last_epoch.json", {}) or {}
dealloc_last = _read_json("deallocation_model_last_epoch.json", {}) or {}

with st.sidebar:
    st.header("Model status")
    st.success("Artifacts loaded")
    st.metric("Allocation inputs", model_input_dim(allocation_model))
    st.metric("Deallocation inputs", model_input_dim(deallocation_model))
    st.metric("Allocation cutoff", f"{float(bundle.get('allocation_cutoff', 0.5)):.3f}")
    st.metric("Max FLMs / row", int(bundle.get("max_units_per_row", 20)))
    st.divider()
    st.caption("Core artifacts")
    for p in REQUIRED:
        st.write(f"✓ `{p}`")

summary_cols = st.columns(4)
summary_cols[0].metric("Allocation Val AUC", f"{float(alloc_last.get('val_auc', 0)):.4f}" if alloc_last else "N/A")
summary_cols[1].metric("Allocation Val Recall", f"{float(alloc_last.get('val_recall', 0)):.4f}" if alloc_last else "N/A")
summary_cols[2].metric("Deallocation Val AUC", f"{float(dealloc_last.get('val_auc', 0)):.4f}" if dealloc_last else "N/A")
summary_cols[3].metric("Deallocation Val Accuracy", f"{float(dealloc_last.get('val_accuracy', 0)):.4f}" if dealloc_last else "N/A")

tab_score, tab_exec, tab_train, tab_features, tab_models, tab_eval, tab_about = st.tabs([
    "Run Models", "Execution Insights", "Training Metrics", "Feature Insight", "Model Architecture", "Evaluation", "Method"
])

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

with tab_score:
    st.subheader("Score a new allocation file")
    st.write("Upload a `.xlsb`, `.xlsx`, `.xls`, or `.csv` allocation file. The app will engineer features, rank FLMs, apply the optimal cutoff, deallocate excess, and return the scored table.")
    uploaded = st.file_uploader("Upload allocation workbook or CSV", type=["xlsb", "xlsx", "xls", "csv"])
    if uploaded is not None:
        df = read_upload(uploaded)
        st.success(f"Loaded {len(df):,} rows from `{uploaded.name}`")
        with st.expander("Input preview", expanded=False):
            st.dataframe(df.head(250), use_container_width=True, height=300)
        with st.spinner("Running Keras allocation ranker, deallocation ranker, and DC safety pass..."):
            result = score_dataframe_core(df, allocation_model, deallocation_model, bundle)
        st.session_state.last_result = result
        st.session_state.last_uploaded_name = uploaded.name
        render_metric_cards(result)
        st.markdown("### Scored output")
        st.dataframe(result, use_container_width=True, height=620)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download scored CSV", result.to_csv(index=False).encode("utf-8"), file_name="allocation_ai_keras_scored.csv", mime="text/csv")
        c2.download_button("Download scored Excel", to_excel_bytes(result), file_name="allocation_ai_keras_scored.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        diag = summarize_scored(result)
        c3.download_button("Download insights CSV", diag.to_csv(index=False).encode("utf-8"), file_name="allocation_ai_execution_insights.csv", mime="text/csv")
    else:
        st.info("Upload a file to run the two-network allocation/deallocation system.")

with tab_exec:
    st.subheader("Execution insights from latest run")
    result = st.session_state.last_result
    if result is None:
        st.info("Run a file on the **Run Models** tab to populate execution insights.")
    else:
        st.caption(f"Latest file: `{st.session_state.last_uploaded_name}`")
        render_metric_cards(result)
        insights = summarize_scored(result)
        st.dataframe(insights, use_container_width=True, height=420)
        if px is not None:
            # Allocation distribution chart.
            plot_df = result.copy()
            plot_df["AI Final Alloc"] = _num(plot_df["AI Final Alloc"])
            plot_df["AI Deallocation Units"] = _num(plot_df["AI Deallocation Units"])
            if "Flag" in plot_df.columns:
                by_flag = plot_df.groupby(plot_df["Flag"].fillna("Blank").astype(str), dropna=False).agg(
                    rows=("AI Final Alloc", "size"),
                    ai_final_alloc=("AI Final Alloc", "sum"),
                    deallocated=("AI Deallocation Units", "sum"),
                ).reset_index().sort_values("ai_final_alloc", ascending=False).head(20)
                fig = px.bar(by_flag, x="Flag", y=["ai_final_alloc", "deallocated"], title="AI allocation and deallocation by Flag", barmode="group")
                st.plotly_chart(fig, use_container_width=True)
            top_cols = [c for c in ["Item", "Description", "Site", "Flag", "Alloc. Rec.", "AI Allocation Before Deallocation", "AI Deallocation Units", "AI Final Alloc", "AI Left DC"] if c in result.columns]
            st.markdown("#### Top allocations")
            st.dataframe(result.assign(_ai=_num(result["AI Final Alloc"])).sort_values("_ai", ascending=False)[top_cols].head(50), use_container_width=True, height=350)
            st.markdown("#### Largest deallocations")
            st.dataframe(result.assign(_de=_num(result["AI Deallocation Units"])).sort_values("_de", ascending=False)[top_cols].head(50), use_container_width=True, height=350)

with tab_train:
    st.subheader("Training metrics")
    st.write("These metrics come from the uploaded Keras training logs and last-epoch JSON files.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Allocation network last epoch")
        st.json(alloc_last or {})
    with c2:
        st.markdown("### Deallocation network last epoch")
        st.json(dealloc_last or {})

    for title, path in [("Allocation network", "allocation_model_keras_epoch_log.csv"), ("Deallocation network", "deallocation_model_keras_epoch_log.csv")]:
        hist = load_csv_if_exists(path)
        if hist is not None and not hist.empty:
            st.markdown(f"### {title} epoch log")
            st.dataframe(hist.tail(30), use_container_width=True)
            if "epoch" in hist.columns:
                for cols, label in [(["loss", "val_loss"], "Loss"), (["auc", "val_auc"], "AUC"), (["precision", "val_precision", "recall", "val_recall"], "Precision / Recall")]:
                    use_cols = [c for c in cols if c in hist.columns]
                    if use_cols:
                        plot_line(hist, use_cols, f"{title}: {label}")

with tab_features:
    st.subheader("Training feature insight")
    nf = list(bundle.get("num_features", []))
    cf = list(bundle.get("cat_features", []))
    transformed = model_input_dim(allocation_model)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numeric features", len(nf))
    c2.metric("Categorical features", len(cf))
    c3.metric("Transformed inputs", transformed)
    c4.metric("Candidate names", len(bundle.get("candidate_names", [])))
    st.markdown("#### Feature families")
    fam = pd.DataFrame({"feature": nf, "family": [feature_family(x) for x in nf]})
    fam_counts = fam.groupby("family").size().reset_index(name="count").sort_values("count", ascending=False)
    if px is not None:
        fig = px.bar(fam_counts, x="family", y="count", title="Numeric engineered features by family")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fam_counts, use_container_width=True)
    st.markdown("#### Numeric features")
    st.dataframe(fam, use_container_width=True, height=350)
    st.markdown("#### Categorical features")
    st.dataframe(pd.DataFrame({"categorical_feature": cf}), use_container_width=True, height=260)
    st.markdown("#### FLM candidates / token logic")
    st.dataframe(pd.DataFrame({"candidate": bundle.get("candidate_names", [])}), use_container_width=True)

with tab_models:
    st.subheader("Model architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Allocation model")
        st.metric("Parameters", f"{allocation_model.count_params():,}")
        st.metric("Input dimension", model_input_dim(allocation_model))
        st.dataframe(model_layer_table(allocation_model), use_container_width=True, height=420)
    with c2:
        st.markdown("### Deallocation model")
        st.metric("Parameters", f"{deallocation_model.count_params():,}")
        st.metric("Input dimension", model_input_dim(deallocation_model))
        st.dataframe(model_layer_table(deallocation_model), use_container_width=True, height=420)

with tab_eval:
    st.subheader("Evaluation and cutoff selection")
    sweep = load_csv_if_exists("allocation_cutoff_sweep.csv")
    if sweep is not None and not sweep.empty:
        st.markdown("### Allocation cutoff sweep")
        st.dataframe(sweep, use_container_width=True)
        if "mae_before_deallocation" in sweep.columns:
            best = sweep.sort_values("mae_before_deallocation").head(1).iloc[0]
            st.success(f"Best cutoff in sweep: {float(best['cutoff']):.3f} with validation MAE {float(best['mae_before_deallocation']):.4f}")
            if px is not None:
                fig = px.line(sweep, x="cutoff", y="mae_before_deallocation", markers=True, title="Validation MAE by allocation cutoff")
                st.plotly_chart(fig, use_container_width=True)
    eval_df = load_csv_if_exists("file_level_evaluation.csv")
    if eval_df is not None and not eval_df.empty:
        st.markdown("### File-level evaluation")
        st.dataframe(eval_df, use_container_width=True)
    preview = load_csv_if_exists("validation_scored_preview.csv")
    if preview is not None and not preview.empty:
        st.markdown("### Validation scored preview")
        st.dataframe(preview.head(500), use_container_width=True, height=400)
    if metrics:
        with st.expander("Raw model_metrics.json", expanded=False):
            st.json(metrics)

with tab_about:
    st.subheader("How this app works")
    st.markdown(
        """
This app uses the rebuilt two-network Keras approach:

1. **Allocation network** — expands each row into possible individual FLM tokens, scores each token, ranks the FLMs, and keeps only FLMs above the optimized cutoff.
2. **Cutoff optimizer** — uses the saved validation sweep to select the cutoff that best matched historical `Final Alloc.` before deallocation.
3. **Deallocation network** — looks only at allocated FLM tokens where the first pass creates DC pressure and removes the least deserving / highest-removal-probability FLMs.
4. **DC safety pass** — guarantees the final `AI Left DC` is never negative.

The app outputs the original dataset plus:

- `AI Starting Left DC`
- `AI Allocation Before Deallocation`
- `AI Deallocation Units`
- `AI DC Safety Cut`
- `AI Final Alloc`
- `AI Left DC`
- `AI Allocated FLMs Before Deallocation`
- `AI Allocation Cutoff`
"""
    )
