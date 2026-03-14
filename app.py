from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import compute_overview_insights, load_datasets
from utils.ui import (
    render_header,
    render_insights,
    render_metric_cards,
    render_persona_flow,
    setup_page,
)

setup_page("Overview | Student Academic Success Platform", route_path="/")

df, subject_df = load_datasets()

render_header(
    "Overview",
    "Institution-wide snapshot for academics, risk, placement readiness, and lifestyle impact.",
)
render_persona_flow()

total_students = len(df)
avg_cgpa = df["mean_cgpa"].mean()
at_risk_students = int(df["at_risk_student"].sum())
placement_readiness = df["placement_readiness_score"].mean()
internship_rate = df["internship_participation"].mean() * 100
avg_study_hours = df["study_hours_day"].mean()

render_metric_cards(
    [
        {"label": "Total Students", "value": f"{total_students}", "subtext": "Unified records"},
        {"label": "Average CGPA", "value": f"{avg_cgpa:.2f}", "subtext": "Institution average"},
        {"label": "At-Risk Students", "value": f"{at_risk_students}", "subtext": "Needs monitoring"},
        {
            "label": "Placement Readiness %",
            "value": f"{placement_readiness:.1f}%",
            "subtext": "Mean readiness score",
        },
        {
            "label": "Internship Participation Rate",
            "value": f"{internship_rate:.1f}%",
            "subtext": "Based on profile indicators",
        },
        {
            "label": "Average Study Hours",
            "value": f"{avg_study_hours:.2f} hrs",
            "subtext": "Per day",
        },
    ]
)

left, right = st.columns(2)

with left:
    st.subheader("1) CGPA Distribution")
    cgpa_fig = px.histogram(
        df,
        x="mean_cgpa",
        nbins=30,
        color_discrete_sequence=["#16A34A"],
        labels={"mean_cgpa": "CGPA"},
    )
    cgpa_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340)
    st.plotly_chart(cgpa_fig, width="stretch")

with right:
    st.subheader("2) Subject Performance Ranking")
    subject_view = subject_df.copy()
    subject_view["avg_marks"] = subject_view["avg_theory_gp"] * 10
    subject_rank = subject_view.sort_values("avg_marks", ascending=False).head(12)
    rank_fig = px.bar(
        subject_rank,
        x="avg_marks",
        y="subject",
        orientation="h",
        color="avg_marks",
        color_continuous_scale="Tealgrn",
        labels={"avg_marks": "Average Marks", "subject": "Subject"},
    )
    rank_fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
    )
    st.plotly_chart(rank_fig, width="stretch")

left_bottom, right_bottom = st.columns(2)

with left_bottom:
    st.subheader("3) Study Hours vs CGPA")
    scatter_fig = px.scatter(
        df,
        x="study_hours_day",
        y="mean_cgpa",
        color="risk_level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        labels={"study_hours_day": "Study Hours / Day", "mean_cgpa": "CGPA"},
        opacity=0.75,
    )
    scatter_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340)
    st.plotly_chart(scatter_fig, width="stretch")

with right_bottom:
    st.subheader("4) Stress Level Distribution")
    stress_counts = (
        df["stress_level"]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Nan": "Unknown"})
        .value_counts()
        .rename_axis("Stress Level")
        .reset_index(name="Students")
    )
    stress_fig = px.bar(
        stress_counts,
        x="Stress Level",
        y="Students",
        color="Stress Level",
        color_discrete_sequence=["#16A34A", "#065F46", "#F59E0B", "#16A34A", "#065F46"],
    )
    stress_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=340)
    st.plotly_chart(stress_fig, width="stretch")

st.subheader("Insights Panel")
render_insights(compute_overview_insights(df, subject_df))

st.info("Use the sidebar to open persona-specific dashboards and workflows.")

