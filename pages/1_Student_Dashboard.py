from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import (
    get_semester_columns,
    get_student_lookup,
    load_datasets,
    predict_student_cgpa,
    train_models_cached,
)
from utils.ui import render_header, render_metric_cards, setup_page

setup_page("Student Dashboard | Student Analytics", route_path="/Student_Dashboard")

df, _ = load_datasets()

render_header(
    "Student Dashboard",
    "Personalized academic, career, and risk analytics for a selected student.",
)

student_lookup = get_student_lookup(df)
selected_student = st.selectbox("Select Student", list(student_lookup.keys()))
student_row = df.loc[student_lookup[selected_student]]

with st.spinner("Running prediction model..."):
    ml_results = train_models_cached(df)
predicted_cgpa = predict_student_cgpa(student_row, ml_results)

render_metric_cards(
    [
        {
            "label": "Current CGPA",
            "value": f"{student_row['latest_cgpa']:.2f}",
            "subtext": "Latest reported",
        },
        {
            "label": "Predicted CGPA",
            "value": f"{predicted_cgpa:.2f}",
            "subtext": "Model estimate",
        },
        {
            "label": "Placement Readiness Score",
            "value": f"{student_row['placement_readiness_score']:.1f}%",
            "subtext": "Career readiness",
        },
        {
            "label": "Startup Potential Score",
            "value": f"{student_row['startup_potential_score']:.1f}",
            "subtext": "Innovation indicator",
        },
        {
            "label": "Risk Level",
            "value": f"{student_row['risk_level']}",
            "subtext": student_row["risk_reason"],
        },
    ]
)

top_left, top_right = st.columns(2)

with top_left:
    st.subheader("1) CGPA Trend per Semester")
    semester_cols = get_semester_columns(df)
    sem_df = pd.DataFrame(
        {
            "Semester": [col.split("_")[-1] for col in semester_cols],
            "SGPA": [student_row[col] for col in semester_cols],
        }
    ).dropna()
    if sem_df.empty:
        st.warning("Semester-wise SGPA data not available for this student.")
    else:
        trend_fig = px.line(
            sem_df,
            x="Semester",
            y="SGPA",
            markers=True,
            color_discrete_sequence=["#16A34A"],
        )
        trend_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(trend_fig, width="stretch")

with top_right:
    st.subheader("2) Study Hours vs Performance")
    study_fig = px.scatter(
        df,
        x="study_hours_day",
        y="mean_cgpa",
        color="risk_level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        opacity=0.65,
        labels={"study_hours_day": "Study Hours / Day", "mean_cgpa": "CGPA"},
    )
    study_fig.add_trace(
        go.Scatter(
            x=[student_row["study_hours_day"]],
            y=[student_row["mean_cgpa"]],
            mode="markers+text",
            marker=dict(size=14, color="#065F46", symbol="diamond"),
            text=["Selected Student"],
            textposition="top center",
            name="Selected Student",
            showlegend=False,
        )
    )
    study_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(study_fig, width="stretch")

bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.subheader("3) Stress vs Exam Score")
    stress_fig = px.scatter(
        df,
        x="stress_numeric",
        y="mean_cgpa",
        color="risk_level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        labels={"stress_numeric": "Stress (0 low - 3 high)", "mean_cgpa": "CGPA"},
        opacity=0.65,
    )
    stress_fig.add_trace(
        go.Scatter(
            x=[student_row["stress_numeric"]],
            y=[student_row["mean_cgpa"]],
            mode="markers+text",
            marker=dict(size=14, color="#065F46", symbol="diamond"),
            text=["Selected Student"],
            textposition="top center",
            showlegend=False,
        )
    )
    stress_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(stress_fig, width="stretch")

with bottom_right:
    st.subheader("Personalized Recommendations")
    recommendations: list[str] = []
    if student_row["study_hours_day"] < 3:
        recommendations.append("Increase study hours to at least 3/day.")
    if student_row["social_hours"] > 2:
        recommendations.append("Reduce social media usage during study windows.")
    if student_row["technical_skill_score"] < 65:
        recommendations.append("Improve programming and technical project depth.")
    if student_row["attendance_rate"] < 0.75:
        recommendations.append("Improve attendance consistency above 75%.")
    if student_row["stress_numeric"] >= 2:
        recommendations.append("Use stress-management routines before exam cycles.")
    if not recommendations:
        recommendations.append("Maintain current performance and start advanced interview prep.")

    for rec in recommendations:
        st.markdown(f"- {rec}")

    st.subheader("Career Recommendation")
    st.success(f"Recommended Career Path: {student_row['career_path']}")
    gaps_value = student_row["skill_gaps"]
    if isinstance(gaps_value, list):
        gaps = gaps_value
    elif isinstance(gaps_value, str):
        gaps = [item.strip() for item in gaps_value.split(",") if item.strip()]
    else:
        gaps = []
    if gaps:
        st.write("Suggested skill focus:")
        for gap in gaps[:3]:
            st.markdown(f"- {gap}")

