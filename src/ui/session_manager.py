"""Streamlit Session State Manager for Multimodal RAG.

Maintains conversation history, active document context, user configurations,
and UI state across Streamlit reruns.
"""

from typing import Any, Dict, List, Optional
import streamlit as st


def initialize_session() -> None:
    """Initialize default state variables in st.session_state if not present."""
    defaults: Dict[str, Any] = {
        "messages": [],
        "active_document": None,
        "uploaded_documents": [],
        "processing_status": {},
        "text_limit": 5,
        "image_limit": 5,
        "max_generated_images": 1,
        "current_diagram": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_message(role: str, content: str, response: Optional[Dict[str, Any]] = None) -> None:
    """Append a user or assistant message to the persistent session history."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "response": response,
    })


def clear_chat() -> None:
    """Clear conversation history."""
    st.session_state.messages = []

