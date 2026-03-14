from __future__ import annotations

import plotly.io as pio
import streamlit as st

BACKGROUND = "#FFFFFF"
PRIMARY = "#16A34A"
SECONDARY = "#065F46"
ACCENT = "#F59E0B"
CARD = "#FFFFFF"

CSS = f"""
<style>
:root {{
  --bg: {BACKGROUND};
  --bg-soft: #F0FDF4;
  --primary: {PRIMARY};
  --secondary: {SECONDARY};
  --accent: {ACCENT};
  --card: {CARD};
}}

[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(circle at 12% 10%, #ECFDF3 0%, transparent 44%),
    radial-gradient(circle at 88% 8%, #F0FDF4 0%, transparent 42%),
    linear-gradient(180deg, #FFFFFF 0%, #F4FFF7 100%) !important;
}}

[data-testid="stHeader"] {{
  background: var(--secondary) !important;
  border-bottom: 1px solid var(--secondary);
  min-height: 3.2rem;
}}

[data-testid="stDecoration"] {{
  background: var(--secondary) !important;
}}

[data-testid="stHeader"] *,
[data-testid="stToolbar"] *,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] svg {{
  color: var(--card) !important;
  fill: var(--card) !important;
  opacity: 1 !important;
}}

[data-testid="stToolbar"],
[data-testid="stToolbar"] > div,
[data-testid="stToolbar"] section,
[data-testid="stToolbar"] [role="button"],
[data-testid="stToolbar"] button {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}

[data-testid="stToolbar"] button:hover,
[data-testid="stToolbar"] [role="button"]:hover {{
  background: rgba(255, 255, 255, 0.18) !important;
  border-radius: 8px !important;
}}

[data-testid="stDeployButton"],
[data-testid="stAppDeployButton"],
[aria-label="Deploy"],
[title="Deploy"] {{
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
  background: transparent !important;
}}

[data-testid="stDeployButton"] > button {{
  background: var(--primary) !important;
  color: var(--card) !important;
  border: 1px solid var(--card) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}}

[data-testid="stDeployButton"] > button:hover {{
  background: var(--accent) !important;
  color: var(--secondary) !important;
  border-color: var(--card) !important;
}}

[data-testid="stAppViewBlockContainer"] {{
  padding-top: 2.0rem;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #FFFFFF 0%, var(--bg-soft) 100%);
  border-right: 1px solid var(--secondary);
}}

[data-testid="stSidebarNav"] {{
  display: none !important;
}}

html,
body,
[class*="css"] {{
  font-family: "Poppins", "Trebuchet MS", "Gill Sans", sans-serif;
  color: var(--secondary) !important;
}}

h1,
h2,
h3,
h4,
h5,
h6 {{
  font-family: "Avenir Next", "Segoe UI Semibold", "Gill Sans", sans-serif;
  letter-spacing: 0.2px;
  color: var(--secondary) !important;
}}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
label,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
.stText,
.stAlert,
.stDataFrame,
.stTable {{
  color: var(--secondary) !important;
}}

div[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {{
  background: var(--card) !important;
  color: var(--secondary) !important;
  border-color: var(--primary) !important;
}}

[data-testid="stSidebarNav"] {{
  padding-top: 0.4rem;
}}

[data-testid="stSidebarNav"] a {{
  background: var(--card) !important;
  border: 1px solid var(--primary) !important;
  border-radius: 10px !important;
  margin: 0.2rem 0 !important;
  padding: 0.35rem 0.45rem !important;
  color: var(--secondary) !important;
}}

[data-testid="stSidebarNav"] a:hover {{
  background: var(--bg) !important;
  border-color: var(--accent) !important;
}}

[data-testid="stSidebarNav"] a[aria-current="page"] {{
  background: var(--primary) !important;
  border-color: var(--primary) !important;
}}

[data-testid="stSidebarNav"] a[aria-current="page"] * {{
  color: var(--card) !important;
}}

[data-testid="stSidebarNav"] a * {{
  color: var(--secondary) !important;
  font-weight: 600 !important;
}}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {{
  color: var(--secondary) !important;
  font-weight: 600 !important;
}}

.stButton > button {{
  background: var(--primary);
  color: var(--card);
  border: 1px solid var(--primary);
  border-radius: 8px;
}}

.stButton > button:hover {{
  background: var(--accent);
  border-color: var(--accent);
  color: var(--secondary);
}}

.metric-card {{
  background: var(--card);
  border: 1px solid var(--primary);
  border-radius: 14px;
  padding: 0.9rem 1rem;
  min-height: 102px;
}}

.metric-card .label {{
  color: var(--secondary);
  font-size: 0.84rem;
  margin-bottom: 0.2rem;
}}

.metric-card .value {{
  color: var(--secondary);
  font-size: 1.8rem;
  font-weight: 700;
  line-height: 1.1;
}}

.metric-card .sub {{
  color: var(--secondary);
  font-size: 0.8rem;
  margin-top: 0.2rem;
}}

.panel-card {{
  background: var(--card);
  border: 1px solid var(--primary);
  border-radius: 14px;
  padding: 0.8rem 1rem;
}}

.insight-pill {{
  background: var(--card);
  border: 1px solid var(--accent);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 0.55rem 0.75rem;
  margin-bottom: 0.5rem;
  color: var(--secondary);
}}

.stTabs [data-baseweb="tab"] {{
  color: var(--secondary);
  border-bottom: 2px solid transparent;
}}

.stTabs [aria-selected="true"] {{
  color: var(--primary);
  border-bottom: 2px solid var(--primary);
}}

[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
  background: var(--card) !important;
  border: 1px solid var(--primary) !important;
  border-radius: 10px !important;
  margin: 0.22rem 0 !important;
  color: var(--secondary) !important;
}}

[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
  border-color: var(--accent) !important;
  background: #F0FDF4 !important;
}}

[data-testid="stSidebar"] [aria-current="page"][data-testid="stPageLink-NavLink"] {{
  background: var(--primary) !important;
  border-color: var(--primary) !important;
}}

[data-testid="stSidebar"] [aria-current="page"][data-testid="stPageLink-NavLink"] * {{
  color: var(--card) !important;
}}
</style>
"""


NAV_ITEMS = [
    ("app.py", "Overview", ":material/home:"),
    ("pages/1_Student_Dashboard.py", "Student Dashboard", ":material/school:"),
    ("pages/2_Faculty_Analytics.py", "Faculty Analytics", ":material/co_present:"),
    ("pages/3_Mentor_Risk_Monitoring.py", "Mentor / Risk Monitoring", ":material/monitor_heart:"),
    ("pages/4_Career_Placement_Insights.py", "Career & Placement Insights", ":material/work:"),
    ("pages/5_Lifestyle_Impact_Analytics.py", "Lifestyle Impact Analytics", ":material/insights:"),
    ("pages/6_AI_Academic_Advisor.py", "AI Academic Advisor", ":material/smart_toy:"),
    (
        "pages/7_Data_Quality_ETL_Monitoring.py",
        "Data Quality & ETL Monitoring",
        ":material/data_check:",
    ),
    ("pages/8_Admin_Panel.py", "Admin Panel", ":material/admin_panel_settings:"),
]


def _render_sidebar_nav() -> None:
    st.sidebar.markdown("### Navigation")
    if hasattr(st, "page_link"):
        for path, label, icon in NAV_ITEMS:
            st.sidebar.page_link(path, label=label, icon=icon)
    else:
        for _, label, icon in NAV_ITEMS:
            st.sidebar.markdown(f"- {icon} {label}")


def setup_page(page_title: str, icon: str = ":bar_chart:", route_path: str = "/") -> None:
    pio.templates.default = "plotly_white"
    st.set_page_config(
        page_title=page_title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("Student Academic Success & Career Analytics Platform")
    st.sidebar.caption("Persona-based dashboards")
    _render_sidebar_nav()
    with st.sidebar.expander("Website Structure", expanded=False):
        st.markdown("\n".join([f"- {label}" for _, label, _ in NAV_ITEMS]))


def render_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def render_persona_flow() -> None:
    st.markdown("### Persona Flow")
    st.caption("Choose a role to jump directly to persona-specific dashboards.")
    cols = st.columns(4)
    if hasattr(st, "page_link"):
        with cols[0]:
            st.page_link("pages/1_Student_Dashboard.py", label="Student", icon=":material/school:")
        with cols[1]:
            st.page_link(
                "pages/2_Faculty_Analytics.py",
                label="Faculty",
                icon=":material/co_present:",
            )
        with cols[2]:
            st.page_link(
                "pages/3_Mentor_Risk_Monitoring.py",
                label="Mentor",
                icon=":material/monitor_heart:",
            )
        with cols[3]:
            st.page_link(
                "pages/8_Admin_Panel.py",
                label="Admin",
                icon=":material/admin_panel_settings:",
            )
    else:
        with cols[0]:
            st.write("Student -> Student Dashboard")
        with cols[1]:
            st.write("Faculty -> Faculty Analytics")
        with cols[2]:
            st.write("Mentor -> Risk Monitoring")
        with cols[3]:
            st.write("Admin -> Admin Panel")


def render_metric_cards(metrics: list[dict[str, str]]) -> None:
    columns = st.columns(len(metrics))
    for col, metric in zip(columns, metrics, strict=False):
        subtext = metric.get("subtext", "")
        col.markdown(
            (
                "<div class='metric-card'>"
                f"<div class='label'>{metric['label']}</div>"
                f"<div class='value'>{metric['value']}</div>"
                f"<div class='sub'>{subtext}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_insights(insights: list[str]) -> None:
    if not insights:
        st.info("No insights available yet.")
        return
    for insight in insights:
        st.markdown(f"<div class='insight-pill'>{insight}</div>", unsafe_allow_html=True)
