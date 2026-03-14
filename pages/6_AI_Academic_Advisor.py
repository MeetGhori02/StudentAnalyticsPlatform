from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import generate_advisor_response, load_datasets
from utils.ui import render_header, setup_page

setup_page("AI Academic Advisor | Student Analytics", route_path="/AI_Academic_Advisor")

df, subject_df = load_datasets()

render_header(
    "AI Academic Advisor",
    "Ask natural-language questions about student risk, subjects, placement, and readiness. Includes rule engine + RAG retrieval.",
)

st.write("Example questions:")
q_col_1, q_col_2, q_col_3 = st.columns(3)
quick_prompt = ""
if q_col_1.button("Which students are at risk?"):
    quick_prompt = "Which students are at risk?"
if q_col_2.button("Which subject is hardest?"):
    quick_prompt = "Which subject is hardest?"
if q_col_3.button("How is placement readiness overall?"):
    quick_prompt = "How is placement readiness overall?"

if "advisor_messages" not in st.session_state:
    st.session_state.advisor_messages = [
        {
            "role": "assistant",
            "content": "Ask me about risk counts, difficult subjects, internships, or student IDs.",
        }
    ]

for message in st.session_state.advisor_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask a question about student performance...")
if quick_prompt:
    prompt = quick_prompt

if prompt:
    st.session_state.advisor_messages.append({"role": "user", "content": prompt})
    response = generate_advisor_response(prompt, df, subject_df)
    st.session_state.advisor_messages.append({"role": "assistant", "content": response})
    st.rerun()
