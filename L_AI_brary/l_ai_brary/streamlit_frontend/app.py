import os
import sys
import streamlit as st

from pydantic import BaseModel
from crewai.flow.flow import Flow, start

from src.l_ai_brary.main import ChatState, ChatbotFlow


# -----------------------------

# Streamlit UI
# To run the app, navigate to the l_ai_brary directory and run:

# python -m streamlit run streamlit_frontend/app.py

# -----------------------------



st.set_page_config(page_title="L_AI_brary", page_icon="📚", layout="wide")
st.title("📚 L_AI_brary Chatbot")

# -----------------------------
# Initialize flow in session state
# -----------------------------
if "chat_flow" not in st.session_state:
    st.session_state.chat_flow = ChatbotFlow()

flow = st.session_state.chat_flow

# -----------------------------
# Show existing conversation
# -----------------------------
for msg in flow.state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Chat input (only if not quit)
# -----------------------------
if not flow.state.user_quit:
    user_msg = st.chat_input("Ask me about a book...")
    if user_msg:
        flow.take_user_input(user_msg)
        st.rerun()

# -----------------------------
# Quit button
# -----------------------------
if not flow.state.user_quit:
    if st.button("Quit Chat"):
        flow.state.user_quit = True
        st.rerun()  # refresh UI immediately

# -----------------------------
# Goodbye message
# -----------------------------
if flow.state.user_quit:
    st.markdown(
        """
        <div style='text-align: center; font-size: 22px; padding: 2em;'>
            😉 Thank you for coming to our <b>L_AI_brary</b>!  
            <br>We hope to see you again soon. 📚✨
        </div>
        """,
        unsafe_allow_html=True
    )