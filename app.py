"""Multimodal RAG Assistant - Streamlit Application (Phase 9).

Integrates:
- PDF document upload, extraction, chunking, embedding, and Qdrant indexing
- Intelligent multimodal request routing (PDF QA, original figure retrieval, new image generation)
- Visual chat interface with original document figures, AI illustrations, and citations
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Automatically load environment variables from .env
load_dotenv(Path(__file__).resolve().parent / ".env")

# Streamlit Page Configuration
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.generation.request_router import RequestRouter
from src.ui.chat_interface import ChatInterface
from src.ui.display_manager import DisplayManager
from src.ui.document_manager import DocumentManager
from src.ui.session_manager import add_message, clear_chat, initialize_session

# Initialize Application State
initialize_session()
document_manager = DocumentManager()
chat_interface = ChatInterface()
display_manager = DisplayManager()
request_router = RequestRouter()


# -----------------------------------------------------------------------------
# SIDEBAR: DOCUMENT & CONFIGURATION MANAGEMENT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Documents & Settings")
    st.caption("Manage PDF collections and retrieval parameters.")

    st.markdown("---")
    st.subheader("1. Upload & Index PDF")
    uploaded_file = st.file_uploader(
        "Select a PDF document",
        type=["pdf"],
        help="Upload enterprise documents, whitepapers, or textbooks with diagrams.",
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index PDF", use_container_width=True):
            try:
                pdf_path = document_manager.save_uploaded_file(uploaded_file)
                info = document_manager.process_pdf(pdf_path)
                st.session_state.active_document = info["document_id"]
                st.success(f"Indexed: **{info['filename']}**")
                st.rerun()
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

    st.markdown("---")
    st.subheader("2. Active Document")
    indexed_docs = document_manager.get_indexed_documents()
    if indexed_docs:
        doc_names = [d["filename"] for d in indexed_docs]
        doc_ids = [d["document_id"] for d in indexed_docs]

        # Determine default index
        default_idx = 0
        if st.session_state.active_document in doc_ids:
            default_idx = doc_ids.index(st.session_state.active_document)

        selected_name = st.selectbox(
            "Target Document for Search & QA",
            options=doc_names,
            index=default_idx,
            help="Queries and retrieval will be filtered to this document.",
        )
        st.session_state.active_document = doc_ids[doc_names.index(selected_name)]
        st.info(f"Active ID: `{st.session_state.active_document}`")
    else:
        st.warning("No documents indexed yet. Upload a PDF above.")
        st.session_state.active_document = None

    st.markdown("---")
    st.subheader("3. Cloud Model Engine")
    cloud_provider = st.selectbox(
        "Provider",
        options=["gemini", "openai"],
        format_func=lambda x: "✨ Google Gemini (Free Tier)" if x == "gemini" else "🤖 OpenAI (GPT-4o)",
        index=0,
        help="Multimodal inference is processed via official cloud endpoints using your .env credentials.",
    )
    env_key_name = f"{cloud_provider.upper()}_API_KEY"
    has_key = bool(os.environ.get(env_key_name))

    if has_key:
        st.success(f"🔐 `{env_key_name}` active from `.env`")
    else:
        st.error(f"❌ `{env_key_name}` missing in `.env`")
        st.caption(f"Add `{env_key_name}=your_key_here` into `.env` to execute queries.")

    st.markdown("---")
    st.subheader("4. Pipeline Parameters")
    text_limit = st.slider("Max Text Chunks", min_value=1, max_value=10, value=5)
    image_limit = st.slider("Max Retrieved Images", min_value=1, max_value=8, value=2)

    operation_mode = st.radio(
        "Operation Mode",
        options=["auto", "qa", "retrieve", "generate"],
        format_func=lambda x: {
            "auto": "🤖 Smart Router (Auto)",
            "qa": "📖 PDF Question Answering Only",
            "retrieve": "🔍 Original PDF Image Retrieval Only",
            "generate": "🎨 AI Image Generation Only",
        }[x],
        index=0,
    )

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        clear_chat()
        st.rerun()


# -----------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------------------------------
st.title("🌌 Multimodal RAG Assistant")
st.markdown(
    "Query your documents with **BGE-M3**, **SigLIP 2**, and **Google Gemini / OpenAI**, "
    "or synthesize new illustrations with **Stable Diffusion Turbo**."
)
st.markdown("---")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("response"):
            display_manager.display_response(msg["response"])
        else:
            st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question about the PDF or request a technical illustration..."):
    # Render and store user message
    with st.chat_message("user"):
        st.markdown(prompt)
    add_message("user", prompt)

    # Route and check PDF requirements
    if operation_mode == "auto":
        route = request_router.route(prompt)
        intent = route["intent"]
    else:
        intent = {
            "qa": "answer",
            "retrieve": "retrieve_pdf_images",
            "generate": "generate_image",
        }[operation_mode]

    needs_pdf = intent in ["answer", "retrieve_pdf_images", "answer_and_generate"]

    if needs_pdf and not st.session_state.active_document:
        warning_msg = (
            "⚠️ Please upload and select an active PDF in the sidebar first. "
            "You can generate a new AI image without an uploaded document."
        )
        with st.chat_message("assistant"):
            st.warning(warning_msg)
        add_message("assistant", warning_msg)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Processing multimodal request..."):
                try:
                    response_payload = chat_interface.ask(
                        query=prompt,
                        document_id=st.session_state.active_document,
                        text_limit=text_limit,
                        image_limit=image_limit,
                        operation=operation_mode,
                        cloud_provider=cloud_provider,
                    )
                    display_manager.display_response(response_payload)
                    add_message(
                        "assistant",
                        content=response_payload.get("answer") or "Visual response generated.",
                        response=response_payload,
                    )
                except Exception as err:
                    st.error(f"Error processing request: {str(err)}")

