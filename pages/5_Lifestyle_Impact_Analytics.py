from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import load_datasets
from utils.ui import render_header, render_insights, render_metric_cards, setup_page

setup_page("Lifestyle Impact Analytics | Student Analytics", route_path="/Lifestyle_Impact_Analytics")

df, _ = load_datasets()

render_header(
    "Lifestyle Impact Analytics",
    "Behavior patterns and wellbeing signals linked with academic performance.",
)

avg_study = df["study_hours_day"].mean()
avg_sleep = df["sleep_hours"].mean()
avg_stress = df["stress_numeric"].mean()
screen_time_corr = df["social_hours"].corr(df["mean_cgpa"])

render_metric_cards(
    [
        {
            "label": "Average Study Hours",
            "value": f"{avg_study:.2f} hrs",
            "subtext": "Per day",
        },
        {
            "label": "Average Sleep Hours",
            "value": f"{avg_sleep:.2f} hrs",
            "subtext": "Estimated daily sleep",
        },
        {
            "label": "Average Stress Level",
            "value": f"{avg_stress:.2f} / 3",
            "subtext": "0 low, 3 high",
        },
        {
            "label": "Screen Time Impact",
            "value": f"{screen_time_corr:.2f}",
            "subtext": "Correlation with CGPA",
        },
    ]
)

row_one_left, row_one_right = st.columns(2)

with row_one_left:
    st.subheader("1) Study Hours vs CGPA")
    study_fig = px.scatter(
        df,
        x="study_hours_day",
        y="mean_cgpa",
        color="risk_level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        labels={"study_hours_day": "Study Hours / Day", "mean_cgpa": "CGPA"},
        opacity=0.7,
    )
    study_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(study_fig, width="stretch")

with row_one_right:
    st.subheader("2) Sleep Hours vs Exam Score")
    sleep_fig = px.scatter(
        df,
        x="sleep_hours",
        y="average_marks",
        color="risk_level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        labels={"sleep_hours": "Sleep Hours", "average_marks": "Exam Score"},
        opacity=0.7,
    )
    sleep_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(sleep_fig, width="stretch")

row_two_left, row_two_right = st.columns(2)

with row_two_left:
    st.subheader("3) Social Media vs CGPA")
    social_fig = px.scatter(
        df,
        x="social_hours",
        y="mean_cgpa",
        color="risk_level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        labels={"social_hours": "Social Media Hours", "mean_cgpa": "CGPA"},
        opacity=0.7,
    )
    social_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(social_fig, width="stretch")

with row_two_right:
    st.subheader("4) Stress vs Academic Performance")
    stress_data = df.copy()
    stress_data["stress_band"] = (
        stress_data["stress_numeric"].fillna(-1).round().astype(int).astype(str)
    )
    stress_fig = px.box(
        stress_data,
        x="stress_band",
        y="mean_cgpa",
        color="stress_band",
        color_discrete_map={
            "-1": "#065F46",
            "0": "#16A34A",
            "1": "#16A34A",
            "2": "#F59E0B",
            "3": "#065F46",
        },
        labels={"stress_band": "Stress (0 low - 3 high)", "mean_cgpa": "CGPA"},
    )
    stress_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(stress_fig, width="stretch")

sleep_band = df[(df["sleep_hours"] >= 7) & (df["sleep_hours"] <= 8)]["mean_cgpa"].mean()
outside_sleep_band = df[(df["sleep_hours"] < 7) | (df["sleep_hours"] > 8)]["mean_cgpa"].mean()
sleep_gain = 0.0
if outside_sleep_band and outside_sleep_band > 0:
    sleep_gain = ((sleep_band - outside_sleep_band) / outside_sleep_band) * 100

low_stress_cgpa = df[df["stress_numeric"] <= 1]["mean_cgpa"].mean()
high_stress_cgpa = df[df["stress_numeric"] >= 2]["mean_cgpa"].mean()
stress_drop = low_stress_cgpa - high_stress_cgpa

render_insights(
    [
        f"Students sleeping 7-8 hours perform about {sleep_gain:.1f}% better on average.",
        f"High stress bands reduce average CGPA by approximately {stress_drop:.2f}.",
    ]
)

