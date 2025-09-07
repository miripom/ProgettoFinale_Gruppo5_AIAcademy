import os
from pathlib import Path
import streamlit as st

from src.l_ai_brary.main import ChatbotFlow
from utils.rag_utils import RAG_Settings, index_pdf_in_qdrant

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

# -----------------------------
# PDF upload and indexing
# -----------------------------
if "indexing_in_progress" not in st.session_state:
    st.session_state.indexing_in_progress = False
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and uploaded_file.name not in st.session_state.indexed_files:
    st.session_state.indexed_files.add(uploaded_file.name)
    st.session_state.indexing_in_progress = True
    knowledge_base_dir = Path(FOLDER_PATH) / "../knowledge_base"
    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = knowledge_base_dir / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"✅ {uploaded_file.name} saved to knowledge_base.")

    flow.state.chat_history.append(
        {"role": "assistant", "content": f"Analyzing the loaded PDF **{uploaded_file.name}**... ⏳"}
    )

    # Run indexing
    with st.spinner("Indexing PDF in Qdrant..."):
        try:
            rag_settings = RAG_Settings()
            index_pdf_in_qdrant(pdf_path=pdf_path, rag_settings=rag_settings)
        except Exception as e:
            st.session_state.indexed_files.remove(uploaded_file.name)
            st.sidebar.error(f"❌ Error indexing {uploaded_file.name}: {e}")
            st.session_state.indexed_files.remove(uploaded_file.name)
            st.session_state.indexing_in_progress = False
            st.stop()

    # Add assistant message: finished
    flow.state.chat_history.append(
        {"role": "assistant", "content": f"✅ PDF **{uploaded_file.name}** fully analyzed and ready!"}
    )
    
    # Mark file as indexed and reset indexing flag
    
st.session_state.indexing_in_progress = False

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
    if not st.session_state.indexing_in_progress:
        user_msg = st.chat_input("Ask me about a book...")
        if user_msg:
            flow.take_user_input(user_msg)
            st.rerun()
    else:
        st.info("Indexing in progress. Please wait...")
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
