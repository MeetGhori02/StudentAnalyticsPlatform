from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import load_datasets, optimize_intervention_plan
from utils.ui import render_header, render_insights, render_metric_cards, setup_page

setup_page("Mentor Risk Monitoring | Student Analytics", route_path="/Mentor_Risk_Monitoring")

df, _ = load_datasets()

render_header(
    "Mentor / Risk Monitoring",
    "Early warning dashboard for advisors to identify and support struggling students quickly.",
)

at_risk_count = int(df["at_risk_student"].sum())
low_attendance_count = int((df["attendance_rate"] < 0.75).sum())
high_stress_count = int((df["stress_numeric"] >= 2).sum())
backlog_rate = (df["total_backlogs"] > 0).mean() * 100

render_metric_cards(
    [
        {
            "label": "At-Risk Student Count",
            "value": f"{at_risk_count}",
            "subtext": "Composite risk rules",
        },
        {
            "label": "Low Attendance Students",
            "value": f"{low_attendance_count}",
            "subtext": "Attendance below 75%",
        },
        {
            "label": "High Stress Students",
            "value": f"{high_stress_count}",
            "subtext": "Stress levels Bad/Awful",
        },
        {"label": "Backlog Rate", "value": f"{backlog_rate:.1f}%", "subtext": "Students with backlog"},
    ]
)

st.subheader("Risk Table")
risk_table = (
    df.loc[df["risk_level"].isin(["High", "Medium"]), ["student_id_perf", "risk_level", "risk_reason"]]
    .sort_values("risk_level", ascending=False)
    .rename(
        columns={
            "student_id_perf": "Student",
            "risk_level": "Risk Level",
            "risk_reason": "Reason",
        }
    )
)
risk_table["Student"] = risk_table["Student"].astype(int).apply(lambda sid: f"Student {sid}")
st.dataframe(risk_table.head(60), width="stretch", hide_index=True)

left, center, right = st.columns(3)

with left:
    st.subheader("1) Risk Level Distribution")
    risk_dist = df["risk_level"].value_counts().rename_axis("Risk Level").reset_index(name="Students")
    pie_fig = px.pie(
        risk_dist,
        names="Risk Level",
        values="Students",
        color="Risk Level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        hole=0.45,
    )
    pie_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    st.plotly_chart(pie_fig, width="stretch")

with center:
    st.subheader("2) Stress vs CGPA")
    stress_fig = px.scatter(
        df,
        x="stress_numeric",
        y="mean_cgpa",
        color="risk_level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        labels={"stress_numeric": "Stress (0 low - 3 high)", "mean_cgpa": "CGPA"},
        opacity=0.7,
    )
    stress_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    st.plotly_chart(stress_fig, width="stretch")

with right:
    st.subheader("3) Attendance vs Performance")
    attendance_fig = px.scatter(
        df,
        x="attendance_rate",
        y="mean_cgpa",
        color="risk_level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        labels={"attendance_rate": "Attendance Rate", "mean_cgpa": "CGPA"},
        opacity=0.7,
    )
    attendance_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    st.plotly_chart(attendance_fig, width="stretch")

st.subheader("Insights")
risk_reason_counts = (
    df.loc[df["at_risk_student"], "risk_reason"]
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
    .value_counts()
)
top_reason = risk_reason_counts.index[0] if not risk_reason_counts.empty else "Composite factors"
high_stress_cgpa = df.loc[df["stress_numeric"] >= 2, "mean_cgpa"].mean()
low_stress_cgpa = df.loc[df["stress_numeric"] <= 1, "mean_cgpa"].mean()
stress_gap = low_stress_cgpa - high_stress_cgpa if pd.notna(high_stress_cgpa) else 0.0
attendance_gap = (
    df.loc[df["attendance_rate"] >= 0.75, "mean_cgpa"].mean()
    - df.loc[df["attendance_rate"] < 0.75, "mean_cgpa"].mean()
)
render_insights(
    [
        f"Top mentor intervention reason is {top_reason.lower()}.",
        f"Students with low attendance (<75%) show around {attendance_gap:.2f} lower CGPA.",
        f"High-stress students average about {stress_gap:.2f} lower CGPA than low-stress peers.",
    ]
)

st.subheader("4) Intervention Optimization")
budget_hours = st.slider(
    "Mentor Hours Budget (per cycle)",
    min_value=10,
    max_value=200,
    value=60,
    step=5,
)
plan_df, plan_summary = optimize_intervention_plan(df, budget_hours=float(budget_hours), max_students=40)
if plan_df.empty:
    st.info("No optimization candidates found for the selected budget.")
else:
    s_col_1, s_col_2, s_col_3 = st.columns(3)
    s_col_1.metric("Selected Students", int(plan_summary["selected_students"]))
    s_col_2.metric("Used Hours", f"{plan_summary['used_hours']:.2f}")
    s_col_3.metric("Projected CGPA Gain", f"{plan_summary['projected_gain']:.2f}")

    st.dataframe(plan_df.head(30), width="stretch", hide_index=True)

    uplift_fig = px.bar(
        plan_df.head(15),
        x="Student",
        y="Projected CGPA Gain",
        color="Risk Level",
        color_discrete_map={"Low": "#16A34A", "Medium": "#F59E0B", "High": "#065F46"},
        labels={"Projected CGPA Gain": "Projected CGPA Gain", "Student": "Student"},
    )
    uplift_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(uplift_fig, width="stretch")

