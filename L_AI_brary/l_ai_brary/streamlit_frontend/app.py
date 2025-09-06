import os
import streamlit as st

from src.l_ai_brary.main import ChatbotFlow

FILE_PATH = os.path.abspath(__file__)       # streamlit_frontend/app.py
FOLDER_PATH = os.path.dirname(FILE_PATH)    # streamlit_frontend

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
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

# Quit button in sidebar
if not flow.state.user_quit:
    if st.sidebar.button("❌ Quit Chat"):
        flow.state.user_quit = True
        st.rerun()  # refresh UI immediately

# PDF uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    knowledge_base_dir = os.path.join(FOLDER_PATH, "../knowledge_base")
    os.makedirs(knowledge_base_dir, exist_ok=True)
    pdf_path = os.path.join(knowledge_base_dir, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"✅ {uploaded_file.name} saved to knowledge_base")

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
