from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import get_semester_columns, load_datasets
from utils.ui import render_header, render_insights, render_metric_cards, setup_page

setup_page("Faculty Analytics | Student Analytics", route_path="/Faculty_Analytics")

df, subject_df = load_datasets()

render_header(
    "Faculty Analytics",
    "Subject-level performance intelligence for teaching strategy and course interventions.",
)

avg_marks_per_subject = (subject_df["avg_theory_gp"] * 10).mean()
failure_rate = subject_df["fail_rate"].mean() * 100
pass_rate = 100 - failure_rate
hardest_subject = subject_df.sort_values("difficulty_score", ascending=False).iloc[0]["subject"]
best_subject = subject_df.sort_values("avg_theory_gp", ascending=False).iloc[0]["subject"]

render_metric_cards(
    [
        {
            "label": "Average Marks per Subject",
            "value": f"{avg_marks_per_subject:.1f}",
            "subtext": "Theory marks equivalent",
        },
        {"label": "Pass Rate", "value": f"{pass_rate:.1f}%", "subtext": "Across subjects"},
        {"label": "Failure Rate", "value": f"{failure_rate:.1f}%", "subtext": "Across subjects"},
        {"label": "Most Difficult Subject", "value": hardest_subject, "subtext": "By risk score"},
        {"label": "Best Performing Subject", "value": best_subject, "subtext": "Highest average GP"},
    ]
)

left, right = st.columns(2)

with left:
    st.subheader("1) Subject Difficulty Heatmap")
    top_difficult = subject_df.sort_values("difficulty_score", ascending=False).head(25)
    heatmap_fig = px.imshow(
        [top_difficult["difficulty_score"].values],
        x=top_difficult["subject"].values,
        y=["Difficulty Score"],
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    heatmap_fig.update_xaxes(tickangle=55)
    heatmap_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(heatmap_fig, width="stretch")

with right:
    st.subheader("2) Average Marks per Subject")
    top_subjects = subject_df.copy()
    top_subjects["avg_marks"] = top_subjects["avg_theory_gp"] * 10
    top_subjects = top_subjects.sort_values("avg_marks", ascending=False).head(15)
    marks_fig = px.bar(
        top_subjects,
        x="avg_marks",
        y="subject",
        orientation="h",
        color="avg_marks",
        color_continuous_scale="Mint",
        labels={"avg_marks": "Average Marks", "subject": "Subject"},
    )
    marks_fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    st.plotly_chart(marks_fig, width="stretch")

st.subheader("3) Class Performance Trend")
sem_cols = get_semester_columns(df)
semester_trend = pd.DataFrame(
    {
        "Semester": [col.split("_")[-1] for col in sem_cols],
        "Average SGPA": [df[col].mean() for col in sem_cols],
    }
).dropna()
trend_fig = px.line(
    semester_trend,
    x="Semester",
    y="Average SGPA",
    markers=True,
    color_discrete_sequence=["#16A34A"],
)
trend_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
st.plotly_chart(trend_fig, width="stretch")

st.subheader("Insights")
most_fail = subject_df.sort_values("fail_rate", ascending=False).iloc[0]
best_improving_sem = semester_trend.sort_values("Average SGPA", ascending=False).iloc[0]["Semester"]
insights = [
    f"{most_fail['subject']} has a {most_fail['fail_rate'] * 100:.1f}% failure rate.",
    f"{best_subject} has the strongest average marks performance.",
    f"Semester {best_improving_sem} has the highest class-level SGPA trend.",
]
render_insights(insights)

