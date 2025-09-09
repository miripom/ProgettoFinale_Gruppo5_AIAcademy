import os
from pathlib import Path
import time
import streamlit as st
import asyncio

from l_ai_brary.main import ChatbotFlow
from l_ai_brary.utils.rag_utils import RAG_Settings, index_pdf_in_qdrant

import threading

def run_flow(flow: ChatbotFlow):
    flow.kickoff()


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

if "crewai_flow" not in st.session_state:
    print("haven't started crewai flow yet")
    st.session_state.crewai_flow = ChatbotFlow()
    threading.Thread(target=run_flow, args=(st.session_state.crewai_flow,), daemon=True).start()
    print("have started crewai flow")

# Add this check right before your chat display loop
if hasattr(st.session_state.crewai_flow.state, 'needs_refresh') and st.session_state.crewai_flow.state.needs_refresh:
    st.session_state.crewai_flow.state.needs_refresh = False
    st.rerun()

# Add auto-refresh mechanism
if "last_message_count" not in st.session_state:
    st.session_state.last_message_count = 0


# Check if chat history has new messages
current_message_count = len(st.session_state.crewai_flow.state.chat_history)
if current_message_count > st.session_state.last_message_count:
    st.session_state.last_message_count = current_message_count
    st.rerun()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

# Quit button in sidebar
if not st.session_state.crewai_flow.state.user_quit:
    if st.sidebar.button("❌ Quit Chat"):
        st.session_state.crewai_flow.state.user_quit = True
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

    st.session_state.crewai_flow.state.chat_history.append(
        {"role": "assistant", "content": f"Analyzing the loaded PDF **{uploaded_file.name}**... ⏳"}
    )

    # Run indexing
    with st.spinner("Indexing PDF in Qdrant..."):
        try:
            rag_settings = RAG_Settings()
            index_pdf_in_qdrant(pdf_path=pdf_path, rag_settings=rag_settings, crewai_flow=st.session_state.crewai_flow)
            # Add assistant message: finished
            """
            st.session_state.crewai_flow.state.chat_history.append(
                {"role": "assistant", "content": f"✅ PDF **{uploaded_file.name}** fully analyzed and ready!"}
            )
            """
        except Exception as e:
            st.session_state.indexed_files.remove(uploaded_file.name)
            st.sidebar.error(f"❌ Error indexing {uploaded_file.name}: {e}")
            st.session_state.indexed_files.remove(uploaded_file.name)
            st.session_state.indexing_in_progress = False
            st.stop()

    
    
    # Mark file as indexed and reset indexing flag
    
st.session_state.indexing_in_progress = False


# -----------------------------
# Show existing conversation
# -----------------------------
# Debug info
st.sidebar.write(f"Messages count: {len(st.session_state.crewai_flow.state.chat_history)}")
st.sidebar.write(f"User input: '{st.session_state.crewai_flow.state.user_input}'")
st.sidebar.write(f"User quit: {st.session_state.crewai_flow.state.user_quit}")

for msg in st.session_state.crewai_flow.state.chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "text":
            st.write(msg["content"])
        elif msg.get("type") == "image":
            st.image(msg["content"])
        elif msg.get("type") == "md_file":  # TODO: change this into an "put the downloadable file in chat"
            st.markdown(msg["content"])
        else:
            # Default case - just display content as text
            st.write(msg["content"])

# -----------------------------
# Chat input (only if not quit)
# -----------------------------
if not st.session_state.crewai_flow.state.user_quit:
    if not st.session_state.indexing_in_progress:
        user_msg = st.chat_input("Ask me about a book...")
        if user_msg:
            st.session_state.crewai_flow.state.user_input = user_msg
            st.rerun()
    else:
        st.info("Indexing in progress. Please wait...")
        st.rerun()

# -----------------------------
# Goodbye message
# -----------------------------
if st.session_state.crewai_flow.state.user_quit:
    st.markdown(
        """
        <div style='text-align: center; font-size: 22px; padding: 2em;'>
            😉 Thank you for coming to our <b>L_AI_brary</b>!  
            <br>We hope to see you again soon. 📚✨
        </div>
        """,
        unsafe_allow_html=True
    )

if not st.session_state.crewai_flow.state.user_quit:
    time.sleep(.1)
    st.rerun()