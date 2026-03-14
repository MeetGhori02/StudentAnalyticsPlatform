from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import get_student_lookup, load_datasets
from utils.ui import render_header, render_insights, render_metric_cards, setup_page

setup_page("Career & Placement Insights | Student Analytics", route_path="/Career_Placement_Insights")

df, _ = load_datasets()


def _normalize_gaps(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


df["skill_gaps"] = df["skill_gaps"].apply(_normalize_gaps)

render_header(
    "Career & Placement Insights",
    "Readiness scoring, career path recommendations, and skill gap visibility.",
)

placement_readiness = df["placement_readiness_score"].mean()
internship_rate = df["internship_participation"].mean() * 100
avg_technical_skill = df["technical_skill_score"].mean()
career_distribution = df["career_path"].value_counts(normalize=True) * 100
top_career_share = career_distribution.iloc[0] if not career_distribution.empty else 0.0

render_metric_cards(
    [
        {
            "label": "Placement Readiness %",
            "value": f"{placement_readiness:.1f}%",
            "subtext": "Institution average",
        },
        {
            "label": "Internship Participation Rate",
            "value": f"{internship_rate:.1f}%",
            "subtext": "Student participation",
        },
        {
            "label": "Average Technical Skill Score",
            "value": f"{avg_technical_skill:.1f}",
            "subtext": "0 to 100 scale",
        },
        {
            "label": "Career Path Distribution",
            "value": f"{top_career_share:.1f}%",
            "subtext": "Top recommended path share",
        },
    ]
)

left, right = st.columns(2)

with left:
    st.subheader("1) Placement Readiness Distribution")
    ready_fig = px.histogram(
        df,
        x="placement_readiness_score",
        nbins=25,
        color_discrete_sequence=["#16A34A"],
        labels={"placement_readiness_score": "Readiness Score"},
    )
    ready_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(ready_fig, width="stretch")

with right:
    st.subheader("2) Skill Gap Analysis")
    exploded = df["skill_gaps"].explode()
    if exploded.empty:
        st.warning("Skill gap information unavailable.")
    else:
        gap_counts = exploded.value_counts().rename_axis("Skill Gap").reset_index(name="Students")
        gap_fig = px.bar(
            gap_counts.head(10),
            x="Students",
            y="Skill Gap",
            orientation="h",
            color="Students",
            color_continuous_scale="Sunset",
        )
        gap_fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
        )
        st.plotly_chart(gap_fig, width="stretch")

st.subheader("3) Career Path Recommendation Distribution")
career_counts = df["career_path"].value_counts().rename_axis("Career Path").reset_index(name="Students")
career_fig = px.pie(
    career_counts,
    names="Career Path",
    values="Students",
    hole=0.45,
    color_discrete_sequence=["#16A34A", "#065F46", "#F59E0B", "#16A34A", "#065F46"],
)
career_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340)
st.plotly_chart(career_fig, width="stretch")

intern_ready = (
    df.groupby("internship_participation")["placement_readiness_score"].mean().to_dict()
)
with_intern = intern_ready.get(1, 0.0)
without_intern = intern_ready.get(0, 0.0)
intern_gap = with_intern - without_intern
top_path = career_counts.iloc[0]["Career Path"] if not career_counts.empty else "N/A"
top_path_pct = (career_counts.iloc[0]["Students"] / len(df) * 100) if len(df) else 0

st.subheader("Insights")
render_insights(
    [
        f"Internship participation is associated with a {intern_gap:.1f}-point readiness increase.",
        f"{top_path} is the most frequent recommendation at {top_path_pct:.1f}% of students.",
        "Students with lower technical skill scores are driving most repeated skill-gap tags.",
    ]
)

st.subheader("Example Output")
student_lookup = get_student_lookup(df)
selected_student = st.selectbox("Select Student for Career Output", list(student_lookup.keys()))
student_row = df.loc[student_lookup[selected_student]]

st.write(f"Student: {int(student_row['student_id_perf'])}")
st.write(f"Placement Readiness: {student_row['placement_readiness_score']:.1f}%")
st.write(f"Suggested Career: {student_row['career_path']}")
st.write("Skill Gaps:")
for gap in student_row["skill_gaps"][:4]:
    st.markdown(f"- {gap}")

