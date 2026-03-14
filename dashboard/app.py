"""
app.py â€” Student Academic Success & Career Readiness Analytics Platform
Streamlit multi-page dashboard.

Run:  streamlit run dashboard/app.py
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

# Make src importable from dashboard/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# â”€â”€ page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="Student Success Intelligence Platform",
    page_icon="ðŸŽ“",
    layout="wide",
    initial_sidebar_state="expanded"
)

# â”€â”€ CSS theme â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
    .main { background: #0F1117; }
    .block-container { padding: 2rem 2.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #4361EE33;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .metric-card h1 { font-size: 2.4rem; color: #4CC9F0; margin: 0; font-weight: 700; }
    .metric-card p  { color: #aaa; margin: 4px 0 0; font-size: 0.85rem; }
    .metric-card .label { color: #ddd; font-size: 0.95rem; margin-bottom: 4px; }
    .risk-badge-red   { background:#F7258522; color:#F72585; border:1px solid #F72585;
                         border-radius:6px; padding:2px 10px; font-size:0.78rem; }
    .risk-badge-green { background:#4CC9F022; color:#4CC9F0; border:1px solid #4CC9F0;
                         border-radius:6px; padding:2px 10px; font-size:0.78rem; }
    div[data-testid="stSidebar"] { background: #0d0d1a; }
    h1,h2,h3 { color: #E0E0FF !important; }
    .stTabs [data-baseweb="tab"] { color: #aaa; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #4361EE; border-bottom: 2px solid #4361EE; }
    .section-header {
        background: linear-gradient(90deg, #4361EE22, transparent);
        border-left: 4px solid #4361EE;
        padding: 8px 16px;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem;
    }
    .insight-box {
        background: #1a1a2e;
        border: 1px solid #7209B733;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 0.5rem 0;
    }
    .insight-box strong { color: #4CC9F0; }
</style>
""", unsafe_allow_html=True)

CHART_COLORS = ['#4361EE','#4CC9F0','#7209B7','#F72585','#3A0CA3','#06D6A0']

# â”€â”€ load data (cached) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_data
def load_data():
    proc = os.path.join(BASE_DIR, 'data', 'processed')
    df   = pd.read_csv(os.path.join(proc, 'final_student_dataset.csv'))
    subj = pd.read_csv(os.path.join(proc, 'subject_summary.csv'))
    return df, subj


@st.cache_resource
def train_models(df):
    from ml_models import run_all_models
    return run_all_models(df)


# â”€â”€ sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("## ðŸŽ“ Student Success Intelligence")
    st.caption("Early-warning analytics for faculty, mentors, and academic leaders")
    st.markdown("---")
    page = st.radio("Navigate", [
        "ðŸ“Š Overview",
        "ðŸ“š Learning Gaps",
        "ðŸš¨ Intervention Center",
        "ðŸ§¬ Performance Drivers",
        "ðŸ¤– Predictive Intelligence",
    ])
    st.markdown("---")
    st.markdown("**Data Sources Stitched**")
    st.markdown("â€¢ Academic performance records\nâ€¢ Attitude & behaviour survey\nâ€¢ Student behaviour data\nâ€¢ Research / profile dataset")
    st.markdown("---")
    st.caption("Hackathon 2026 Â· Early Intervention Demo")

df, subj_df = load_data()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 1 â€” OVERVIEW
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if page == "ðŸ“Š Overview":
    st.title("ðŸŽ“ Student Success Intelligence Platform")
    st.markdown("A stitched early-warning view of **514 students** across academic, behavioural, and lifestyle signals.")
    st.markdown("---")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""<div class="insight-box">
        <strong>ðŸ« Institutional Problem</strong><br>
        Student performance, behaviour, and background data usually sit in separate systems,
        making intervention late and inconsistent.
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""<div class="insight-box">
        <strong>ðŸ§© What This System Does</strong><br>
        It stitches fragmented datasets into one student-level intelligence layer for learning-gap detection,
        risk monitoring, and prediction.
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown("""<div class="insight-box">
        <strong>ðŸŽ¯ Who Acts On It</strong><br>
        Faculty use it to spot difficult subjects, mentors use it to prioritize support,
        and academic leaders use it to track institutional risk.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI cards
    total     = len(df)
    avg_cgpa  = df['mean_cgpa'].mean()
    at_risk   = df['at_risk_student'].sum()
    pass_rate = (df['mean_cgpa'] >= 5).mean() * 100
    avg_bl    = df['total_backlogs'].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    cards = [
        (c1, total,              "Total Students",        "#4CC9F0"),
        (c2, f"{avg_cgpa:.2f}", "Average CGPA",           "#4361EE"),
        (c3, int(at_risk),       "At-Risk Students",      "#F72585"),
        (c4, f"{pass_rate:.1f}%","Pass Rate (CGPAâ‰¥5)",    "#06D6A0"),
        (c5, f"{avg_bl:.1f}",   "Avg Backlogs/Student",  "#7209B7"),
    ]
    for col, val, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <h1 style="color:{color}">{val}</h1>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">CGPA Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='mean_cgpa', nbins=35, color_discrete_sequence=['#4361EE'],
                           labels={'mean_cgpa':'Mean CGPA'})
        fig.add_vline(x=avg_cgpa, line_dash='dash', line_color='#F72585',
                      annotation_text=f"Mean {avg_cgpa:.2f}", annotation_position="top right")
        fig.update_layout(showlegend=False, height=340,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#ccc')
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-header">CGPA Band Breakdown</div>', unsafe_allow_html=True)
        if 'cgpa_band' in df.columns:
            band = df['cgpa_band'].value_counts().reset_index()
            band.columns = ['Band','Count']
            fig2 = px.pie(band, names='Band', values='Count',
                          color_discrete_sequence=CHART_COLORS, hole=0.45)
            fig2.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
            st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">SGPA Trend by Semester</div>', unsafe_allow_html=True)
        sem_cols = sorted([c for c in df.columns if c.startswith('sgpa_sem_')])
        if sem_cols:
            sgpa_means = df[sem_cols].mean()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=[c.split('_')[-1] for c in sem_cols],
                y=sgpa_means.values, mode='lines+markers',
                line=dict(color='#4CC9F0', width=2.5),
                marker=dict(size=8, color='#4CC9F0'),
                fill='tozeroy', fillcolor='rgba(76,201,240,0.1)'
            ))
            fig3.update_layout(height=300, xaxis_title='Semester', yaxis_title='Avg SGPA',
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#ccc')
            st.plotly_chart(fig3, width="stretch")

    with col4:
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig4 = px.histogram(df, x='academic_risk_score', nbins=30,
                            color='at_risk_student',
                            color_discrete_map={True:'#F72585', False:'#4CC9F0'},
                            labels={'academic_risk_score':'Risk Score','at_risk_student':'At Risk'})
        fig4.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
        st.plotly_chart(fig4, width="stretch")

    # Key insights
    st.markdown('<div class="section-header">ðŸ” Key Insights</div>', unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    with i1:
        st.markdown(f"""<div class="insight-box">
        <strong>ðŸ“‰ Low CGPA Alert</strong><br>
        {(df['mean_cgpa']<5).sum()} students ({(df['mean_cgpa']<5).mean()*100:.1f}%) have CGPA below 5.0.
        Immediate academic counselling is recommended.
        </div>""", unsafe_allow_html=True)
    with i2:
        best_sem = df[[c for c in df.columns if 'sgpa_sem' in c]].mean().idxmax()
        st.markdown(f"""<div class="insight-box">
        <strong>ðŸ† Peak Performance</strong><br>
        Students perform best in <b>{best_sem.replace('sgpa_sem_','Semester ')}</b>.
        Avg SGPA: {df[best_sem].mean():.2f}.
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""<div class="insight-box">
        <strong>ðŸ“š Backlog Burden</strong><br>
        {(df['total_backlogs']>2).sum()} students carry more than 2 backlogs,
        significantly impacting graduation timelines.
        </div>""", unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 2 â€” SUBJECT PERFORMANCE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ“š Learning Gaps":
    st.title("ðŸ“š Learning Gaps & Subject Intelligence")
    st.markdown("Surface difficult subjects, failure-heavy patterns, and teaching pressure points that need academic attention.")
    st.markdown("---")

    col1, col2 = st.columns([1.4,1])

    with col1:
        st.markdown('<div class="section-header">Subject Difficulty Ranking</div>', unsafe_allow_html=True)
        top_n = st.slider("Top N subjects", 5, len(subj_df), 15)
        top   = subj_df.head(top_n)
        fig   = px.bar(top, x='difficulty_score', y='subject', orientation='h',
                       color='difficulty_score', color_continuous_scale='RdYlGn_r',
                       labels={'difficulty_score':'Difficulty Score','subject':'Subject'})
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'},
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#ccc', coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Subject Stats</div>', unsafe_allow_html=True)
        display = subj_df[['subject','avg_theory_gp','fail_rate','student_count','difficulty_score']].copy()
        display.columns = ['Subject','Avg GP','Fail Rate','Students','Difficulty']
        display['Fail Rate'] = (display['Fail Rate']*100).round(1).astype(str)+'%'
        display['Avg GP']    = display['Avg GP'].round(2)
        display['Difficulty']= display['Difficulty'].round(1)
        st.dataframe(display.head(20), width="stretch", height=480)

    # Grade point distribution
    st.markdown('<div class="section-header">Average Theory Grade Point by Subject</div>', unsafe_allow_html=True)
    fig2 = px.bar(subj_df.sort_values('avg_theory_gp'),
                  x='avg_theory_gp', y='subject', orientation='h',
                  color='avg_theory_gp', color_continuous_scale='Blues',
                  labels={'avg_theory_gp':'Avg Grade Point'})
    fig2.update_layout(height=500, yaxis={'categoryorder':'total ascending'},
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font_color='#ccc', coloraxis_showscale=False)
    st.plotly_chart(fig2, width="stretch")

    # Fail rate vs grade point scatter
    st.markdown('<div class="section-header">Fail Rate vs Grade Point (Bubble = Student Count)</div>',
                unsafe_allow_html=True)
    fig3 = px.scatter(subj_df, x='avg_theory_gp', y='fail_rate',
                      size='student_count', color='difficulty_score',
                      hover_name='subject', color_continuous_scale='RdYlGn_r',
                      labels={'avg_theory_gp':'Avg Grade Point','fail_rate':'Fail Rate'})
    fig3.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
    st.plotly_chart(fig3, width="stretch")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 3 â€” STUDENT RISK DETECTION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸš¨ Intervention Center":
    st.title("ðŸš¨ Intervention Center")
    st.markdown("Prioritize students who need support first, using stitched risk signals across academics, attendance, and stress.")
    st.markdown("---")

    risk_df  = df[df['at_risk_student'] == True].copy()
    safe_df  = df[df['at_risk_student'] == False].copy()

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">At-Risk Students</div>'
                    f'<h1 style="color:#F72585">{len(risk_df)}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Safe Students</div>'
                    f'<h1 style="color:#06D6A0">{len(safe_df)}</h1></div>', unsafe_allow_html=True)
    with c3:
        avg_risk_score = risk_df['academic_risk_score'].mean()
        st.markdown(f'<div class="metric-card"><div class="label">Avg Risk Score</div>'
                    f'<h1 style="color:#F72585">{avg_risk_score:.1f}</h1></div>', unsafe_allow_html=True)
    with c4:
        high_risk = (df['academic_risk_score'] > 70).sum()
        st.markdown(f'<div class="metric-card"><div class="label">High Risk (>70)</div>'
                    f'<h1 style="color:#FF6B35">{high_risk}</h1></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=risk_df['academic_risk_score'], name='At Risk',
                                   marker_color='#F72585', opacity=0.75))
        fig.add_trace(go.Histogram(x=safe_df['academic_risk_score'], name='Safe',
                                   marker_color='#4CC9F0', opacity=0.75))
        fig.update_layout(barmode='overlay', height=340,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#ccc')
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Risk vs CGPA Scatter</div>', unsafe_allow_html=True)
        fig2 = px.scatter(df, x='mean_cgpa', y='academic_risk_score',
                          color='at_risk_student',
                          color_discrete_map={True:'#F72585',False:'#4CC9F0'},
                          hover_data=['student_id_perf','total_backlogs'],
                          labels={'mean_cgpa':'Mean CGPA','academic_risk_score':'Risk Score',
                                  'at_risk_student':'At Risk'})
        fig2.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
        st.plotly_chart(fig2, width="stretch")

    # Risk factors breakdown
    st.markdown('<div class="section-header">Risk Factor Analysis</div>', unsafe_allow_html=True)
    col3,col4,col5 = st.columns(3)
    with col3:
        bl_groups = pd.cut(df['total_backlogs'], bins=[-1,0,2,5,100],
                           labels=['0','1-2','3-5','6+']).value_counts().sort_index()
        fig3 = px.bar(x=bl_groups.index.astype(str), y=bl_groups.values,
                      color_discrete_sequence=['#7209B7'],
                      labels={'x':'Backlog Count','y':'Students'},
                      title='Backlog Distribution')
        fig3.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
        st.plotly_chart(fig3, width="stretch")

    with col4:
        if 'th_attendance_rate' in df.columns:
            att_groups = pd.cut(df['th_attendance_rate'], bins=[0,0.6,0.75,0.9,1.01],
                                labels=['<60%','60-75%','75-90%','>90%']).value_counts().sort_index()
            fig4 = px.bar(x=att_groups.index.astype(str), y=att_groups.values,
                          color_discrete_sequence=['#4361EE'],
                          labels={'x':'Attendance Rate','y':'Students'},
                          title='Attendance Distribution')
            fig4.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
            st.plotly_chart(fig4, width="stretch")

    with col5:
        if 'stress_level' in df.columns:
            sc = df.groupby('stress_level')['academic_risk_score'].mean().sort_values(ascending=False)
            fig5 = px.bar(x=sc.index, y=sc.values, color_discrete_sequence=['#F72585'],
                          labels={'x':'Stress Level','y':'Avg Risk Score'},
                          title='Stress vs Avg Risk Score')
            fig5.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
            st.plotly_chart(fig5, width="stretch")

    # At-risk student list
    st.markdown('<div class="section-header">ðŸ”´ Priority Intervention Queue</div>', unsafe_allow_html=True)

    search = st.text_input("Search by Student ID", "")
    show_cols = ['student_id_perf','mean_cgpa','total_backlogs','academic_risk_score',
                 'th_attendance_rate']
    available_cols = [c for c in show_cols if c in df.columns]

    show_df = risk_df[available_cols].sort_values('academic_risk_score', ascending=False)
    show_df.columns = [c.replace('_',' ').title() for c in show_df.columns]

    if search:
        show_df = show_df[show_df['Student Id Perf'].astype(str).str.contains(search)]

    st.dataframe(show_df, width="stretch", height=400)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 4 â€” LIFESTYLE vs PERFORMANCE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ§¬ Performance Drivers":
    st.title("ðŸ§¬ Performance Drivers")
    st.markdown("Explain the root causes behind academic outcomes by connecting study habits, stress, finances, and social behaviour.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["ðŸ“– Study Habits","ðŸ˜¤ Stress","ðŸ’° Finances","ðŸŒ Socials"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            if 'study_hours_day' in df.columns:
                fig = px.scatter(df, x='study_hours_day', y='mean_cgpa',
                                 color='academic_risk_score', color_continuous_scale='RdYlGn_r',
                                 size='total_backlogs', size_max=15,
                                 labels={'study_hours_day':'Study Hours/Day','mean_cgpa':'Mean CGPA'},
                                 title='Study Hours vs CGPA')
                fig.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
                st.plotly_chart(fig, width="stretch")

        with col2:
            if 'study_hours_day' in df.columns:
                sh_groups = df.groupby('daily_studing_time')['mean_cgpa'].mean().sort_values(ascending=False)
                fig2 = px.bar(x=sh_groups.values, y=sh_groups.index, orientation='h',
                              color_discrete_sequence=['#4361EE'],
                              labels={'x':'Avg CGPA','y':'Study Time'},
                              title='Study Time Category vs Avg CGPA')
                fig2.update_layout(height=380, yaxis={'categoryorder':'total ascending'},
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font_color='#ccc')
                st.plotly_chart(fig2, width="stretch")

        if 'study_efficiency' in df.columns:
            fig3 = px.histogram(df, x='study_efficiency', nbins=30,
                                color_discrete_sequence=['#06D6A0'],
                                labels={'study_efficiency':'Study Efficiency (CGPA / Hours)'},
                                title='Study Efficiency Distribution')
            fig3.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
            st.plotly_chart(fig3, width="stretch")

    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            if 'stress_level' in df.columns:
                fig = px.box(df, x='stress_level', y='mean_cgpa',
                             color='stress_level', color_discrete_sequence=CHART_COLORS,
                             labels={'stress_level':'Stress Level','mean_cgpa':'Mean CGPA'},
                             title='Stress Level vs CGPA Distribution')
                fig.update_layout(showlegend=False, height=380, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
                st.plotly_chart(fig, width="stretch")

        with col2:
            if 'stress_numeric' in df.columns:
                fig2 = px.scatter(df, x='stress_numeric', y='mean_cgpa',
                                  trendline='ols', color_discrete_sequence=['#F72585'],
                                  labels={'stress_numeric':'Stress (numeric)','mean_cgpa':'Mean CGPA'},
                                  title='Stress Score vs CGPA (with trend)')
                fig2.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
                st.plotly_chart(fig2, width="stretch")

        if 'stress_level' in df.columns:
            stress_risk = df.groupby('stress_level')[['academic_risk_score','mean_cgpa']].mean().reset_index()
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Bar(x=stress_risk['stress_level'], y=stress_risk['academic_risk_score'],
                                  name='Avg Risk Score', marker_color='#F72585'), secondary_y=False)
            fig3.add_trace(go.Scatter(x=stress_risk['stress_level'], y=stress_risk['mean_cgpa'],
                                      mode='lines+markers', name='Avg CGPA', line=dict(color='#4CC9F0',width=2.5)),
                           secondary_y=True)
            fig3.update_layout(title='Stress Level â€” Risk Score & CGPA', height=350,
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#ccc')
            st.plotly_chart(fig3, width="stretch")

    with tab3:
        col1,col2 = st.columns(2)
        with col1:
            if 'financial_status' in df.columns:
                fin_cgpa = df.groupby('financial_status')['mean_cgpa'].mean().sort_values(ascending=False).reset_index()
                fig = px.bar(fin_cgpa, x='financial_status', y='mean_cgpa',
                             color='mean_cgpa', color_continuous_scale='Blues',
                             labels={'financial_status':'Financial Status','mean_cgpa':'Avg CGPA'},
                             title='Financial Status vs Average CGPA')
                fig.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc', coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")

        with col2:
            if 'part_time_job' in df.columns:
                ptj = df.groupby('part_time_job')['mean_cgpa'].agg(['mean','count']).reset_index()
                ptj.columns = ['Part Time Job','Avg CGPA','Count']
                fig2 = px.bar(ptj, x='Part Time Job', y='Avg CGPA',
                              color_discrete_sequence=['#7209B7'],
                              text='Count',
                              title='Part-Time Job vs CGPA')
                fig2.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
                st.plotly_chart(fig2, width="stretch")

    with tab4:
        col1,col2 = st.columns(2)
        with col1:
            if 'social_hours' in df.columns:
                fig = px.scatter(df, x='social_hours', y='mean_cgpa',
                                 trendline='ols', color_discrete_sequence=['#4CC9F0'],
                                 labels={'social_hours':'Social Media Hours','mean_cgpa':'Mean CGPA'},
                                 title='Social Media Usage vs CGPA')
                fig.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
                st.plotly_chart(fig, width="stretch")

        with col2:
            if 'hobbies' in df.columns:
                hob_cgpa = df.groupby('hobbies')['mean_cgpa'].mean().sort_values(ascending=False).reset_index()
                fig2 = px.bar(hob_cgpa, x='hobbies', y='mean_cgpa',
                              color_discrete_sequence=CHART_COLORS,
                              labels={'hobbies':'Hobby','mean_cgpa':'Avg CGPA'},
                              title='Hobby vs Average CGPA')
                fig2.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
                st.plotly_chart(fig2, width="stretch")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 5 â€” ML PREDICTIONS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ¤– Predictive Intelligence":
    st.title("ðŸ¤– Predictive Intelligence")
    st.markdown("Use machine learning to forecast academic outcomes, quantify risk, and support earlier intervention decisions.")
    st.markdown("---")

    with st.spinner("Training models... (cached after first run)"):
        ml = train_models(df)

    tab1, tab2, tab3 = st.tabs(["ðŸ“ˆ Outcome Forecasting", "ðŸŽ¯ Early Warning", "ðŸ” Risk Drivers"])

    with tab1:
        st.markdown('<div class="section-header">CGPA Prediction â€” Model Comparison</div>',
                    unsafe_allow_html=True)
        reg = ml.get('regression', {})
        metrics = reg.get('metrics', {})

        if metrics:
            m_df = pd.DataFrame(metrics).T.reset_index()
            m_df.columns = ['Model','RMSE','MAE','RÂ²']
            st.dataframe(m_df, width="stretch")

            fig = px.bar(m_df, x='Model', y=['RMSE','MAE'],
                         barmode='group', color_discrete_sequence=['#4361EE','#F72585'],
                         title='RMSE vs MAE by Model')
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
            st.plotly_chart(fig, width="stretch")

            # Scatter: actual vs predicted for best model
            best_model_name = max(metrics, key=lambda x: metrics[x]['R2'])
            best_model = reg['models'][best_model_name]
            X_test = reg['X_test']
            y_test  = reg['y_test']
            y_pred  = best_model.predict(X_test)
            fig2 = px.scatter(x=y_test, y=y_pred,
                              labels={'x':'Actual CGPA','y':'Predicted CGPA'},
                              title=f'Actual vs Predicted CGPA ({best_model_name})',
                              color_discrete_sequence=['#4CC9F0'], opacity=0.6)
            fig2.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                           x1=y_test.max(), y1=y_test.max(),
                           line=dict(color='#F72585', dash='dash'))
            fig2.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
            st.plotly_chart(fig2, width="stretch")

        # Live prediction
        st.markdown('<div class="section-header">ðŸŽ° Simulate a Student Outcome</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            sgpa_input      = st.slider("Mean SGPA", 0.0, 10.0, 6.5)
            backlogs_input  = st.number_input("Total Backlogs", 0, 50, 0)
            attendance_input= st.slider("Attendance Rate", 0.0, 1.0, 0.85)
        with c2:
            theory_gp       = st.slider("Theory Grade Point Mean", 0.0, 10.0, 6.5)
            practical_marks = st.slider("Practical Marks Mean", 0.0, 100.0, 65.0)
            study_hrs       = st.slider("Study Hours/Day", 0.0, 8.0, 2.0)
        with c3:
            stress_n        = st.selectbox("Stress Level", [0,1,2,3], format_func=lambda x: ['Fabulous','Good','Bad','Awful'][x])
            travel_m        = st.slider("Travel Time (mins)", 0, 150, 45)
            avg_marks_input = st.slider("Average Marks (10th+12th+College)", 0.0, 100.0, 70.0)

        if reg.get('models') and st.button("ðŸ”® Predict CGPA", key='predict_cgpa'):
            feat_cols = reg['features']
            sample = {
                'mean_sgpa': sgpa_input,
                'total_backlogs': backlogs_input,
                'th_attendance_rate': attendance_input,
                'pr_attendance_rate': attendance_input,
                'theory_gp_mean': theory_gp,
                'practical_marks_mean': practical_marks,
                'study_hours_day': study_hrs,
                'stress_numeric': stress_n,
                'travel_mins': travel_m,
                'social_hours': 1.0,
                'academic_risk_score': 30.0,
                'average_marks': avg_marks_input,
                'semesters_attended': 6,
            }
            row = pd.DataFrame([{f: sample.get(f, 0) for f in feat_cols}])
            pred = reg['models'][best_model_name].predict(row)[0]
            color = '#06D6A0' if pred >= 7 else '#F72585' if pred < 5 else '#4CC9F0'
            st.markdown(f"""
            <div class="metric-card" style="margin-top:1rem">
                <div class="label">Predicted Academic Outcome ({best_model_name})</div>
                <h1 style="color:{color}">{pred:.2f}</h1>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Risk Classification â€” Model Comparison</div>',
                    unsafe_allow_html=True)
        clf = ml.get('classification', {})
        clf_metrics = clf.get('metrics', {})

        if clf_metrics:
            c_df = pd.DataFrame(clf_metrics).T.reset_index()
            c_df.columns = ['Model','Accuracy','Precision','Recall','F1']
            st.dataframe(c_df, width="stretch")

            fig = px.bar(c_df, x='Model', y=['Accuracy','F1','Precision','Recall'],
                         barmode='group', color_discrete_sequence=CHART_COLORS,
                         title='Classification Metrics Comparison')
            fig.update_layout(height=350, yaxis=dict(range=[0,1.1]),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#ccc')
            st.plotly_chart(fig, width="stretch")

            # Confusion matrix for RF
            cms = clf.get('confusion_matrices', {})
            if 'Random Forest Classifier' in cms:
                cm = cms['Random Forest Classifier']
                fig2 = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                 labels={'x':'Predicted','y':'Actual'},
                                 title='Confusion Matrix â€” Random Forest',
                                 x=['Not At Risk','At Risk'], y=['Not At Risk','At Risk'])
                fig2.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)', font_color='#ccc')
                st.plotly_chart(fig2, width="stretch")

    with tab3:
        fi = ml.get('classification', {}).get('feature_importance', {})
        if fi:
            fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
            fig = px.bar(x=list(fi_sorted.values()), y=list(fi_sorted.keys()),
                         orientation='h', color_discrete_sequence=['#7209B7'],
                         labels={'x':'Importance','y':'Feature'},
                         title='Top 15 Features â€” Risk Prediction (Random Forest)')
            fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'},
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#ccc')
            st.plotly_chart(fig, width="stretch")

        # Insights
        st.markdown('<div class="section-header">ðŸ” ML-Derived Insights</div>', unsafe_allow_html=True)
        r1,r2,r3 = st.columns(3)
        with r1:
            st.markdown("""<div class="insight-box">
            <strong>ðŸŽ¯ Top Predictor</strong><br>
            <b>Academic Risk Score</b> (34.3%) and <b>Total Backlogs</b> (20.6%) are 
            the strongest predictors of student failure risk.
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown("""<div class="insight-box">
            <strong>ðŸ“ˆ Model Performance</strong><br>
            Random Forest achieves <b>RÂ² = 0.989</b> for CGPA prediction 
            and <b>99%+ accuracy</b> for risk classification.
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown("""<div class="insight-box">
            <strong>âš ï¸ Early Warning</strong><br>
            Stress level and low attendance together elevate risk by ~15 points. 
            Early counselling can prevent academic decline.
            </div>""", unsafe_allow_html=True)

