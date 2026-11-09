"""Unit tests for Phase 9 UI components."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import streamlit as st

from src.ui.session_manager import initialize_session, add_message, clear_chat
from src.ui.document_manager import DocumentManager
from src.ui.chat_interface import ChatInterface
from src.ui.display_manager import DisplayManager


class TestSessionManager:
    """Tests for Streamlit session state management."""

    def test_initialize_session(self):
        initialize_session()
        assert "messages" in st.session_state
        assert "active_document" in st.session_state
        assert isinstance(st.session_state.messages, list)

    def test_add_and_clear_messages(self):
        clear_chat()
        assert len(st.session_state.messages) == 0

        add_message("user", "Hello RAG")
        assert len(st.session_state.messages) == 1
        assert st.session_state.messages[0]["content"] == "Hello RAG"
        assert st.session_state.messages[0]["role"] == "user"

        add_message("assistant", "Hello User", response={"answer": "Hello User"})
        assert len(st.session_state.messages) == 2
        assert st.session_state.messages[1]["response"]["answer"] == "Hello User"

        clear_chat()
        assert len(st.session_state.messages) == 0


class TestDocumentManager:
    """Tests for document discovery and path management."""

    def test_get_indexed_documents(self):
        dm = DocumentManager()
        docs = dm.get_indexed_documents()
        assert isinstance(docs, list)
        if docs:
            assert "document_id" in docs[0]
            assert "filename" in docs[0]


class TestDisplayManager:
    """Tests for display path resolution."""

    def test_resolve_image_path(self):
        dm = DisplayManager()
        # Non-existent relative path returns as Path object
        p = dm.resolve_image_path("some/fake/path.png")
        assert isinstance(p, Path)


class TestChatInterface:
    """Tests for chat interface configuration."""

    def test_initialization(self):
        ci = ChatInterface()
        assert ci.root.exists()

