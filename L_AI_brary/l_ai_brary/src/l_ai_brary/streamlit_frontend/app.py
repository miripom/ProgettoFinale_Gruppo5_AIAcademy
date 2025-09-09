"""L_AI_brary Streamlit Frontend Application.

This module provides a Streamlit-based web interface for the L_AI_brary chatbot system.
The application allows users to interact with the AI chatbot, upload PDF documents for
indexing into the knowledge base, and manage conversation flow through an intuitive
web interface.

Features:
    - Interactive chat interface with the L_AI_brary chatbot
    - PDF upload and automatic indexing into Qdrant vector database
    - Real-time conversation updates and auto-refresh
    - Session state management for persistent chat history
    - Background processing for PDF indexing
    - Clean exit functionality

Usage:
    To run the application, navigate to the l_ai_brary directory and execute:
    python -m streamlit run streamlit_frontend/app.py
"""

import os
from pathlib import Path
import time
import streamlit as st

from l_ai_brary.main import ChatbotFlow
from l_ai_brary.utils.rag_utils import RAG_Settings, index_pdf_in_qdrant

import threading

def run_flow(flow: ChatbotFlow):
    """Execute the ChatbotFlow in a separate thread.
    
    This function starts the CrewAI flow execution in a background thread,
    allowing the Streamlit interface to remain responsive while the chatbot
    processes user interactions and manages conversation flow.
    
    Args:
        flow (ChatbotFlow): The initialized ChatbotFlow instance to execute.
        
    Note:
        This function is designed to be run in a daemon thread to avoid
        blocking the main Streamlit application thread.
    """
    flow.kickoff()


# =============================================================================
# Application Configuration and Path Setup
# =============================================================================
FILE_PATH = os.path.abspath(__file__)       # streamlit_frontend/app.py
FOLDER_PATH = os.path.dirname(FILE_PATH)    # streamlit_frontend

# =============================================================================
# Streamlit UI Configuration and Initialization
# =============================================================================
# To run the app, navigate to the l_ai_brary directory and run:
# python -m streamlit run streamlit_frontend/app.py

st.set_page_config(page_title="L_AI_brary", page_icon="📚", layout="wide")
st.title("📚 L_AI_brary Chatbot")

# =============================================================================
# CrewAI Flow Initialization and Session State Management
# =============================================================================

# Initialize session state variables first
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = 0

if "crewai_flow" not in st.session_state:
    print("🚀 Initializing CrewAI flow for the first time...")
    st.session_state.crewai_flow = ChatbotFlow()
    st.session_state.flow_initialized = False
    print("✅ CrewAI flow instance created")

# Start the flow only once
if not st.session_state.get("flow_initialized", False):
    print("🔄 Starting CrewAI flow thread...")
    threading.Thread(target=run_flow, args=(st.session_state.crewai_flow,), daemon=True).start()
    st.session_state.flow_initialized = True
    print("✅ CrewAI flow thread started")


# Add this check right before your chat display loop - with rate limiting
if (hasattr(st.session_state.crewai_flow.state, 'needs_refresh') and 
    st.session_state.crewai_flow.state.needs_refresh):
    current_time = time.time()
    if current_time - st.session_state.last_refresh_time > 0.5:  # Rate limit refreshes
        st.session_state.crewai_flow.state.needs_refresh = False
        st.session_state.last_refresh_time = current_time
        st.rerun()

# Add auto-refresh mechanism with rate limiting
if "last_message_count" not in st.session_state:
    st.session_state.last_message_count = 0

# Check if chat history has new messages (with rate limiting)
current_time = time.time()
current_message_count = len(st.session_state.crewai_flow.state.chat_history)
if (current_message_count > st.session_state.last_message_count and 
    current_time - st.session_state.last_refresh_time > 0.5):  # Reduced from 1.0 to 0.5 for faster updates
    st.session_state.last_message_count = current_message_count
    st.session_state.last_refresh_time = current_time
    st.rerun()

# =============================================================================
# Sidebar Controls and User Interface
# =============================================================================
st.sidebar.header("Controls")

# Quit button in sidebar
if not st.session_state.crewai_flow.state.user_quit:
    if st.sidebar.button("❌ Quit Chat"):
        st.session_state.crewai_flow.state.user_quit = True
        st.rerun()  # refresh UI immediately


# =============================================================================
# PDF Upload and Knowledge Base Indexing
# =============================================================================
if "indexing_in_progress" not in st.session_state:
    st.session_state.indexing_in_progress = False
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = ""

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
            # 🔥 Start indexing in background thread
            threading.Thread(
                target=index_pdf_in_qdrant,
                args=(pdf_path, rag_settings, st.session_state.crewai_flow),
                daemon=True
            ).start()
            # index_pdf_in_qdrant(pdf_path=pdf_path, rag_settings=rag_settings, crewai_flow=st.session_state.crewai_flow)
            # Add assistant message: finished
            """
            st.session_state.crewai_flow.state.chat_history.append(
                {"role": "assistant", "content": f"✅ PDF **{uploaded_file.name}** fully analyzed and ready!"}
            )
            """
        except Exception as e:
            st.session_state.indexed_files.discard(uploaded_file.name)  # Use discard to avoid KeyError
            st.sidebar.error(f"❌ Error indexing {uploaded_file.name}: {e}")
            st.session_state.indexing_in_progress = False
            st.stop()

    
    
    # Mark file as indexed and reset indexing flag
    
st.session_state.indexing_in_progress = False


# =============================================================================
# Chat History Display and Message Rendering
# =============================================================================

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

# =============================================================================
# User Input Handling and Chat Interface
# =============================================================================
if not st.session_state.crewai_flow.state.user_quit:
    if not st.session_state.indexing_in_progress:
        user_msg = st.chat_input("Ask me about a book...")
        if user_msg and user_msg.strip():  # Ensure non-empty message
            # Check if this is the same message to avoid duplicate processing
            if user_msg != st.session_state.get("last_user_message", ""):
                # ✅ IMMEDIATELY add user message to chat history for instant display
                st.session_state.crewai_flow.state.chat_history.append({
                    "role": "user", 
                    "content": user_msg
                })
                st.session_state.crewai_flow.state.messages_count += 1
                
                # Set the input for the flow to process
                st.session_state.crewai_flow.state.user_input = user_msg
                st.session_state.last_user_message = user_msg
                st.rerun()
    else:
        st.info("Indexing in progress. Please wait...")
        # Don't auto-rerun during indexing to reduce overhead

# =============================================================================
# Application Exit and Cleanup
# =============================================================================
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
    st.stop()  # Stop execution instead of continuous rerun

# Only refresh if there's active conversation and not quitting
if (not st.session_state.crewai_flow.state.user_quit and 
    len(st.session_state.crewai_flow.state.chat_history) > 0):
    # Use a much longer delay and only refresh occasionally
    time.sleep(2.0)  # Increased from 0.1 to 2.0 seconds
    st.rerun()
