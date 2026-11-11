"""Multimodal RAG Assistant - Streamlit Application.

Integrates:
- PDF document upload, extraction, chunking, embedding, and Qdrant indexing
- Intelligent multimodal request routing (PDF QA, original figure retrieval, new image generation)
- Visual chat interface with original document figures, AI illustrations, and citations
- AI-Powered Diagram & Flowchart Generator grounded in PDF context using Mermaid.js
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components

# Automatically load environment variables from .env
load_dotenv(Path(__file__).resolve().parent / ".env")

# Streamlit Page Configuration
st.set_page_config(
    page_title="Multimodal RAG Assistant & Diagram Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.generation.request_router import RequestRouter
from src.ui.chat_interface import ChatInterface
from src.ui.diagram_interface import DiagramInterface
from src.ui.display_manager import DisplayManager
from src.ui.document_manager import DocumentManager
from src.ui.session_manager import add_message, clear_chat, initialize_session
from src.diagram_generator.diagram_utils import build_mermaid_html

# Initialize Application State
initialize_session()
document_manager = DocumentManager()
chat_interface = ChatInterface()
diagram_interface = DiagramInterface()
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
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------
st.title("🌌 Multimodal RAG Assistant")
st.markdown(
    "Query your documents with **BGE-M3**, **SigLIP 2**, and **Google Gemini / OpenAI**, "
    "or generate interactive technical diagrams using **Mermaid.js**."
)
st.markdown("---")

tab_chat, tab_diagram = st.tabs(["💬 Multimodal Chat & QA", "📊 AI Diagram Generator"])

# =============================================================================
# TAB 1: MULTIMODAL CHAT & QA
# =============================================================================
with tab_chat:
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


# =============================================================================
# TAB 2: AI DIAGRAM GENERATOR (MERMAID.JS)
# =============================================================================
with tab_diagram:
    st.subheader("📊 AI-Powered Diagram & Flowchart Generator")
    st.caption(
        "Generate technically accurate, grounded diagrams directly from your uploaded PDF document. "
        "Rendered client-side using **Mermaid.js** with interactive pan/zoom, SVG, PNG, and .mmd code export."
    )

    if not st.session_state.active_document:
        st.warning("⚠️ No active document selected. Please upload and select a PDF in the sidebar first.")
    else:
        st.success(f"📄 Target Document: **{st.session_state.active_document}**")

    diag_query = st.text_area(
        "Diagram Request / Topic",
        placeholder="e.g. Generate a detailed flowchart illustrating the difference between routing and forwarding, including data plane and control plane operations.",
        height=90,
    )

    col_type, col_dir, col_chunks = st.columns([2, 1, 1])

    with col_type:
        diagram_type_choice = st.selectbox(
            "Diagram Type",
            options=[
                ("auto", "🤖 Automatic (LLM Decides)"),
                ("flowchart", "🔀 Flowchart (flowchart TD/LR)"),
                ("mindmap", "🧠 Mind Map (mindmap)"),
                ("sequenceDiagram", "⏱️ Sequence Diagram (sequenceDiagram)"),
                ("classDiagram", "🏛️ Class Diagram (classDiagram)"),
                ("erDiagram", "🗄️ Entity Relationship Diagram (erDiagram)"),
                ("stateDiagram", "🔄 State Diagram (stateDiagram-v2)"),
                ("timeline", "⏳ Timeline (timeline)"),
                ("architecture", "🏗️ Architecture Diagram (architecture-beta)"),
            ],
            format_func=lambda x: x[1],
            index=0,
            help="Select a specific Mermaid syntax or let the LLM automatically decide based on the content.",
        )
        selected_diagram_type = diagram_type_choice[0]

    with col_dir:
        direction_choice = st.selectbox(
            "Direction",
            options=["TD", "LR"],
            format_func=lambda x: "Top to bottom (TD)" if x == "TD" else "Left to right (LR)",
            index=0,
            help="Used for flowcharts and state diagrams.",
        )

    with col_chunks:
        diag_chunks_limit = st.slider(
            "Context Depth",
            min_value=2,
            max_value=12,
            value=6,
            help="Number of retrieved PDF chunks to supply as ground truth for diagram generation.",
        )

    col_gen, col_clear = st.columns([1, 4])
    with col_gen:
        generate_clicked = st.button("⚡ Generate Diagram", type="primary", use_container_width=True)
    with col_clear:
        if st.session_state.get("current_diagram"):
            if st.button("🗑️ Clear Current Diagram"):
                st.session_state.current_diagram = None
                st.rerun()

    if generate_clicked:
        if not st.session_state.active_document:
            st.error("Please upload and select a PDF document first.")
        elif not diag_query.strip():
            st.error("Please enter a diagram request or topic description.")
        else:
            with st.spinner("Retrieving document context and generating validated Mermaid diagram..."):
                try:
                    result = diagram_interface.generate(
                        query=diag_query.strip(),
                        document_id=st.session_state.active_document,
                        diagram_type=selected_diagram_type,
                        direction=direction_choice,
                        text_limit=diag_chunks_limit,
                        cloud_provider=cloud_provider,
                    )
                    st.session_state.current_diagram = {
                        "query": diag_query.strip(),
                        "document_id": st.session_state.active_document,
                        "type": result.get("diagram_type", selected_diagram_type),
                        "mermaid_code": result.get("mermaid_code", ""),
                        "citations": result.get("citations", []),
                        "context_count": result.get("context_text_count", 0),
                    }
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error generating diagram: {str(ex)}")

    # Display Generated Diagram
    current_diag = st.session_state.get("current_diagram")
    if current_diag and current_diag.get("mermaid_code"):
        mermaid_code = current_diag["mermaid_code"]

        st.markdown("---")
        st.subheader("🖼️ Generated Diagram")

        # Metadata banner
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown(f"**Type:** `{current_diag.get('type')}`")
        with mcol2:
            st.markdown(f"**Source Document:** `{current_diag.get('document_id')}`")
        with mcol3:
            st.markdown(f"**Grounding Chunks:** `{current_diag.get('context_count', 0)} chunks`")

        # Interactive Render via Mermaid.js
        diagram_html = build_mermaid_html(mermaid_code, height=620)
        components.html(diagram_html, height=670, scrolling=True)

        # Download / Export Actions
        st.markdown("#### 💾 Export Diagram")
        dl_col1, dl_col2 = st.columns([1, 3])
        with dl_col1:
            st.download_button(
                label="📥 Download Mermaid Code (.mmd)",
                data=mermaid_code,
                file_name=f"{current_diag.get('document_id', 'diagram')}_{current_diag.get('type', 'diagram')}.mmd",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_col2:
            st.caption("💡 **Tip:** High-resolution SVG and PNG exports (rendered via client-side Canvas), plus Pan & Zoom controls, are built directly into the diagram viewer toolbar above!")

        # Expandable Mermaid Syntax
        with st.expander("📝 View Mermaid Source Code", expanded=False):
            st.code(mermaid_code, language="mermaid")

        # Expandable Grounded Citations
        citations = current_diag.get("citations", [])
        if citations:
            with st.expander(f"📚 Grounded PDF Context & Citations ({len(citations)} pages)", expanded=False):
                for idx, cite in enumerate(citations):
                    page_num = cite.get("page_number", "Unknown")
                    doc_id = cite.get("document_id", "")
                    chunk_cnt = cite.get("chunk_count", 1)
                    preview = cite.get("preview", "")
                    st.markdown(f"**Source {idx+1}:** Document `{doc_id}` — Page **{page_num}** ({chunk_cnt} chunks)")
                    if preview:
                        st.caption(f"> \"{preview}\"")
                    st.markdown("---")
