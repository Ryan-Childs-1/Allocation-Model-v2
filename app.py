from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

from base_features import add_features, read_xlsb
from model_utils import load_numpy_model, score_dataframe_core

APP_TITLE = "Allocation AI — FLM Ranker, TensorFlow-Free"
REQUIRED = ["allocation_model.npz", "deallocation_model.npz", "preprocessing_pipeline.joblib"]
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
    if missing:
        return None, None, None, missing
    allocation_model = load_numpy_model("allocation_model.npz")
    deallocation_model = load_numpy_model("deallocation_model.npz")
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
        try:
            return pd.read_excel(uploaded, sheet_name="3.3 Working Table", header=1)
        except Exception:
            return pd.read_excel(uploaded, header=0)
    return pd.read_csv(uploaded)


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
    if "Final Alloc." in result.columns:
        actual_raw = pd.to_numeric(result["Final Alloc."], errors="coerce")
        mask = actual_raw.notna()
        if mask.any():
            actual = actual_raw.fillna(0)
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
    if n.startswith("t_"):
        return "Per-FLM token features"
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
st.caption("Two separate neural networks exported from Keras to NumPy weights: allocation FLM ranker + deallocation FLM remover. No TensorFlow install required.")

allocation_model, deallocation_model, bundle, missing = load_artifacts()
if missing:
    st.error("The app is missing required runtime artifacts.")
    for m in missing:
        st.code(m)
    st.stop()

metrics = _read_json("model_metrics.json", {})
alloc_last = _read_json("allocation_model_last_epoch.json", {})
dealloc_last = _read_json("deallocation_model_last_epoch.json", {})

with st.sidebar:
    st.header("Model runtime")
    st.success("TensorFlow-free NumPy inference")
    st.metric("Allocation input dim", allocation_model.input_shape[-1])
    st.metric("Deallocation input dim", deallocation_model.input_shape[-1])
    st.metric("Allocation cutoff", bundle.get("allocation_cutoff", "not saved"))
    st.metric("Max FLMs per row", bundle.get("max_units_per_row", 20))
    st.divider()
    st.caption("Runtime files")
    for f in REQUIRED:
        st.write(f"✅ `{f}`")

tabs = st.tabs(["Run Models", "Execution Insights", "Training Metrics", "Feature Insight", "Model Architecture", "Evaluation", "Method"])

if "scored_result" not in st.session_state:
    st.session_state.scored_result = None

with tabs[0]:
    st.subheader("Run allocation and deallocation models")
    uploaded = st.file_uploader("Upload allocation workbook or CSV", type=["xlsb", "xlsx", "xls", "csv"])
    if uploaded:
        df = read_upload(uploaded)
        st.success(f"Loaded {len(df):,} rows from {uploaded.name}")
        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)
        if st.button("Score file", type="primary"):
            with st.spinner("Scoring FLM tokens, ranking allocations, and correcting DC..."):
                result = score_dataframe_core(df, allocation_model, deallocation_model, bundle)
            st.session_state.scored_result = result
            st.success("Scoring complete.")
    if st.session_state.scored_result is not None:
        result = st.session_state.scored_result
        render_metric_cards(result)
        st.dataframe(result, use_container_width=True, height=550)
        c1, c2 = st.columns(2)
        c1.download_button("Download scored CSV", result.to_csv(index=False).encode("utf-8"), "allocation_ai_scored.csv", "text/csv")
        c2.download_button("Download scored Excel", to_excel_bytes(result), "allocation_ai_scored.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[1]:
    st.subheader("Model execution insights")
    result = st.session_state.scored_result
    if result is None:
        st.info("Run a file on the Run Models tab to populate execution insights.")
    else:
        render_metric_cards(result)
        st.dataframe(summarize_scored(result), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top AI allocations")
            cols = [c for c in ["Item", "Site", "Flag", "Description", "AI Final Alloc", "AI Allocation Before Deallocation", "AI Left DC", "Alloc. Rec."] if c in result.columns]
            st.dataframe(result.sort_values("AI Final Alloc", ascending=False)[cols].head(50), use_container_width=True)
        with c2:
            st.markdown("#### Top deallocations / safety cuts")
            cols = [c for c in ["Item", "Site", "Flag", "Description", "AI Deallocation Units", "AI DC Safety Cut", "AI Final Alloc", "AI Left DC"] if c in result.columns]
            temp = result.assign(_cut=_num(result.get("AI Deallocation Units", pd.Series(0))) + _num(result.get("AI DC Safety Cut", pd.Series(0))))
            st.dataframe(temp.sort_values("_cut", ascending=False)[cols].head(50), use_container_width=True)
        if px is not None:
            chart = summarize_scored(result)
            numeric_rows = chart[pd.to_numeric(chart["Value"], errors="coerce").notna()].copy()
            numeric_rows["Value"] = pd.to_numeric(numeric_rows["Value"], errors="coerce")
            fig = px.bar(numeric_rows.head(12), x="Value", y="Insight", orientation="h", title="Execution summary")
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Training metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Allocation last epoch")
        st.json(alloc_last or {"status": "not available"})
    with c2:
        st.markdown("#### Deallocation last epoch")
        st.json(dealloc_last or {"status": "not available"})
    for label, path in METRIC_FILES.items():
        dfm = load_csv_if_exists(path)
        if dfm is not None:
            with st.expander(label, expanded=label in ["Allocation epoch log", "Deallocation epoch log"]):
                st.dataframe(dfm, use_container_width=True)
                metric_cols = [c for c in dfm.columns if c.startswith("val_") and pd.api.types.is_numeric_dtype(dfm[c])]
                if "epoch" in dfm.columns and metric_cols:
                    plot_line(dfm, metric_cols[:4], f"{label} validation metrics")

with tabs[3]:
    st.subheader("Feature insight")
    num_features = bundle.get("num_features", [])
    cat_features = bundle.get("cat_features", [])
    st.write(f"The deployed preprocessing bundle uses **{len(num_features)} numeric/token features** and **{len(cat_features)} categorical features** before one-hot expansion.")
    feature_df = pd.DataFrame({"feature": num_features, "family": [feature_family(f) for f in num_features]})
    family_counts = feature_df.groupby("family", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(family_counts, use_container_width=True, hide_index=True)
    with c2:
        if px is not None and not family_counts.empty:
            fig = px.bar(family_counts, x="count", y="family", orientation="h", title="Feature families")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Numeric/token features")
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    st.markdown("#### Categorical features")
    st.dataframe(pd.DataFrame({"categorical_feature": cat_features}), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Model architecture")
    for label, model in [("Allocation model", allocation_model), ("Deallocation model", deallocation_model)]:
        info = model.summary_dict()
        st.markdown(f"#### {label}: `{info['name']}`")
        c1, c2 = st.columns(2)
        c1.metric("Input features", info["input_dim"])
        c2.metric("Parameters", f"{info['parameter_count']:,}")
        st.dataframe(pd.DataFrame(info["layers"]), use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Evaluation artifacts")
    if metrics:
        st.markdown("#### model_metrics.json")
        st.json(metrics)
    for label, path in [
        ("Cutoff sweep", "allocation_cutoff_sweep.csv"),
        ("File-level evaluation", "file_level_evaluation.csv"),
        ("Validation scored preview", "validation_scored_preview.csv"),
    ]:
        dfm = load_csv_if_exists(path)
        if dfm is not None:
            with st.expander(label, expanded=True):
                st.dataframe(dfm, use_container_width=True)

with tabs[6]:
    st.subheader("Method")
    st.markdown("""
This deployment uses the two-network FLM-ranking method without installing TensorFlow:

1. **Allocation network** scores every possible FLM token for each row. A row can receive FLM 1, FLM 2, FLM 3, and so on only when the token score clears the saved cutoff.
2. **Cutoff rule** selects how far down the ranked FLM list the app should allocate.
3. **Deallocation network** scores allocated FLM tokens for removal when the first-pass result exceeds item/DC availability.
4. **Final DC safety pass** removes any remaining excess so `AI Left DC` always stays zero or positive.

The original Keras weights were exported into `.npz` files and are evaluated with NumPy. This keeps the same trained networks but avoids TensorFlow installation failures on Streamlit Cloud.
""")
