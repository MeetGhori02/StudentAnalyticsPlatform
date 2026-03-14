from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import get_data_layers_status, get_etl_status, load_datasets
from utils.ui import render_header, render_insights, render_metric_cards, setup_page

setup_page("Data Quality & ETL Monitoring | Student Analytics", route_path="/Data_Quality_ETL_Monitoring")

df, _ = load_datasets()

render_header(
    "Data Quality & ETL Monitoring",
    "Pipeline health, data quality signals, and ETL freshness checks.",
)


def _to_hashable(value: object) -> object:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return str(sorted(value.items()))
    if isinstance(value, set):
        return str(sorted(value))
    return value


hashable_df = df.copy()
for col in hashable_df.columns:
    if hashable_df[col].dtype == object:
        hashable_df[col] = hashable_df[col].map(_to_hashable)

total_cells = df.shape[0] * df.shape[1]
missing_cells = int(df.isna().sum().sum())
missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
duplicates = int(hashable_df.duplicated().sum())
completeness = 100 - missing_pct

render_metric_cards(
    [
        {
            "label": "Total Records Processed",
            "value": f"{len(df)}",
            "subtext": "Rows in processed dataset",
        },
        {"label": "Missing Data %", "value": f"{missing_pct:.2f}%", "subtext": "Across all columns"},
        {"label": "Duplicate Records", "value": f"{duplicates}", "subtext": "Exact row duplicates"},
        {
            "label": "Data Completeness Score",
            "value": f"{completeness:.2f}%",
            "subtext": "100 - missing ratio",
        },
    ]
)

left, right = st.columns(2)

with left:
    st.subheader("1) Missing Value Heatmap")
    missing_by_col = df.isna().mean().sort_values(ascending=False)
    top_missing_cols = missing_by_col[missing_by_col > 0].head(12).index.tolist()
    if not top_missing_cols:
        st.success("No missing values detected.")
    else:
        sampled = df[top_missing_cols].head(120).isna().astype(int)
        heatmap_fig = px.imshow(
            sampled.T,
            color_continuous_scale="Blues",
            labels={"x": "Sampled Rows", "y": "Columns", "color": "Missing"},
            aspect="auto",
        )
        heatmap_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(heatmap_fig, width="stretch")

with right:
    st.subheader("2) Data Distribution per Column")
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        st.warning("No numeric columns available for distribution analysis.")
    else:
        default_idx = numeric_cols.index("mean_cgpa") if "mean_cgpa" in numeric_cols else 0
        selected_col = st.selectbox("Select Numeric Column", numeric_cols, index=default_idx)
        dist_fig = px.histogram(
            df,
            x=selected_col,
            nbins=30,
            color_discrete_sequence=["#16A34A"],
            labels={selected_col: selected_col.replace("_", " ").title()},
        )
        dist_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(dist_fig, width="stretch")

st.subheader("Column Quality Snapshot")
column_quality = (
    pd.DataFrame(
        {
            "Column": df.columns,
            "Missing %": (df.isna().mean() * 100).round(2),
            "Unique Values": [hashable_df[col].nunique(dropna=True) for col in df.columns],
        }
    )
    .sort_values("Missing %", ascending=False)
    .reset_index(drop=True)
)
st.dataframe(column_quality.head(20), width="stretch", hide_index=True)

st.subheader("ETL Logs")
status = get_etl_status()
st.code(
    "\n".join(
        [
            f"ETL Last Run: {status['last_run']}",
            f"New Records Added: {status['new_records']}",
            f"Errors: {status['errors']}",
            f"Status: {status.get('status', 'Unknown')}",
        ]
    )
)

st.subheader("Medallion Layer Status")
layers = get_data_layers_status()
layer_cols = st.columns(3)
layer_cols[0].metric("Bronze Layer Files", layers["bronze"]["files"], layers["bronze"]["last_update"])
layer_cols[1].metric("Silver Layer Files", layers["silver"]["files"], layers["silver"]["last_update"])
layer_cols[2].metric("Gold Layer Files", layers["gold"]["files"], layers["gold"]["last_update"])

st.subheader("Insights")
top_missing_col = (
    column_quality.loc[column_quality["Missing %"].idxmax(), "Column"]
    if not column_quality.empty
    else "N/A"
)
render_insights(
    [
        f"Overall completeness is {completeness:.2f}% across {len(df.columns)} columns.",
        f"Top column with missing values: {top_missing_col}.",
        f"Detected {duplicates} duplicate rows after hash-safe normalization.",
    ]
)

