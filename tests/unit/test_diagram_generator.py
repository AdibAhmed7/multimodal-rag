"""Unit tests for the AI Diagram Generator module."""

from unittest.mock import MagicMock
import pytest

from src.diagram_generator.diagram_prompts import (
    DiagramPromptBuilder,
)
from src.diagram_generator.diagram_service import generate_diagram
from src.diagram_generator.diagram_utils import build_mermaid_html, get_local_mermaid_js
from src.diagram_generator.mermaid_validator import (
    detect_diagram_type,
    extract_mermaid_code,
    sanitize_mermaid_labels,
    validate_mermaid_syntax,
)


class TestMermaidValidator:
    """Tests for Mermaid syntax extraction, label sanitization, and validation."""

    def test_extract_mermaid_code_fenced(self):
        raw = "Here is your diagram:\n```mermaid\nflowchart TD\n  A --> B\n```\nHope that helps!"
        extracted = extract_mermaid_code(raw)
        assert extracted == "flowchart TD\n  A --> B"

    def test_extract_mermaid_code_plain(self):
        raw = "flowchart LR\n  Start[Start] --> End[End]"
        extracted = extract_mermaid_code(raw)
        assert extracted == "flowchart LR\n  Start[Start] --> End[End]"

    def test_sanitize_mermaid_labels_parentheses(self):
        raw = "flowchart TD\n  A[Forwarding (Data Plane)] --> B[Routing (Control Plane)]"
        sanitized = sanitize_mermaid_labels(raw)
        assert 'A["Forwarding (Data Plane)"]' in sanitized
        assert 'B["Routing (Control Plane)"]' in sanitized

    def test_detect_diagram_type(self):
        assert detect_diagram_type("flowchart TD\n  A --> B") == "flowchart"
        assert detect_diagram_type("sequenceDiagram\n  A->>B: hi") == "sequencediagram"
        assert detect_diagram_type("mindmap\n  root((Idea))") == "mindmap"

    def test_validate_valid_flowcharts(self):
        code = "flowchart TD\n  A --> B\n  B --> C"
        valid, err = validate_mermaid_syntax(code)
        assert valid is True
        assert err is None

    def test_validate_valid_mindmap(self):
        code = "mindmap\n  root((Cloud))\n    IaaS\n    PaaS\n    SaaS"
        valid, err = validate_mermaid_syntax(code)
        assert valid is True
        assert err is None

    def test_validate_valid_sequence_diagram(self):
        code = "sequenceDiagram\n  Client->>Server: SYN\n  Server-->>Client: SYN-ACK"
        valid, err = validate_mermaid_syntax(code)
        assert valid is True
        assert err is None

    def test_validate_unsupported_header(self):
        code = "python\nprint('hello world')"
        valid, err = validate_mermaid_syntax(code)
        assert valid is False
        assert "Invalid Mermaid declaration" in err

    def test_validate_unbalanced_brackets(self):
        code = "flowchart TD\n  A[Unclosed bracket --> B"
        valid, err = validate_mermaid_syntax(code)
        assert valid is False
        assert "Unbalanced bracket delimiters" in err

    def test_validate_script_injection(self):
        bad_input = "flowchart TD\n  A[Clean] --> B[<script>alert(1)</script>]"
        valid, err = validate_mermaid_syntax(bad_input)
        assert valid is False
        assert "Disallowed script" in err


class TestDiagramPrompts:
    """Tests for prompt templates and prompt builders."""

    def test_format_diagram_generation_prompt(self):
        context = [
            {
                "document_id": "test_doc",
                "page_number": 1,
                "text_chunks": [{"chunk_id": "c1", "content": "Physical, Data Link, Network..."}],
            }
        ]
        prompt = DiagramPromptBuilder.build_generation_prompt(
            user_query="Explain OSI model",
            diagram_type="flowchart",
            direction="TD",
            pages_context=context,
        )
        assert "Explain OSI model" in prompt
        assert "Physical, Data Link" in prompt
        assert "flowchart TD" in prompt

    def test_format_syntax_repair_prompt(self):
        repair_prompt = DiagramPromptBuilder.build_correction_prompt(
            original_query="Show process",
            failed_mermaid_code="flowchart TD\n A[Open (Paren] --> B",
            error_message="Unbalanced bracket found",
            pages_context=[],
        )
        assert "Unbalanced bracket found" in repair_prompt
        assert "flowchart TD" in repair_prompt


class TestDiagramUtils:
    """Tests for HTML generation and bundled asset discovery."""

    def test_local_mermaid_js_presence(self):
        js = get_local_mermaid_js()
        assert len(js) > 100000  # Local bundle should be > 100KB

    def test_build_mermaid_html(self):
        code = "flowchart LR\n  A --> B"
        html = build_mermaid_html(code)
        assert "<!DOCTYPE html>" in html
        assert "mermaid.initialize" in html
        assert "downloadSVG" in html
        assert "downloadPNG" in html
        assert "flowchart LR" in html


class TestDiagramService:
    """Tests for end-to-end diagram generation service with mock LLM."""

    def test_generate_diagram_success(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "```mermaid\nflowchart TD\n  A[Source] --> B[Destination]\n```"

        result = generate_diagram(
            user_query="Show data flow",
            retrieved_context="Data flows from Source to Destination.",
            diagram_type="flowchart",
            direction="TD",
            llm=mock_llm,
        )

        assert result["success"] is True
        assert result["diagram_type"] == "flowchart"
        assert "A[Source] --> B[Destination]" in result["mermaid_code"]
        assert result["error"] is None

    def test_generate_diagram_with_repair(self):
        mock_llm = MagicMock()
        # First attempt returns invalid header, second attempt corrects it
        mock_llm.generate.side_effect = [
            "invalid_header\n  A --> B",
            "```mermaid\nflowchart TD\n  A --> B\n```",
        ]

        result = generate_diagram(
            user_query="Show process",
            retrieved_context="Step A leads to Step B.",
            diagram_type="flowchart",
            llm=mock_llm,
        )

        assert result["success"] is True
        assert result["diagram_type"] == "flowchart"
        assert result["repair_attempts"] == 1
        assert "flowchart TD" in result["mermaid_code"]

    def test_generate_diagram_insufficient_info(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "INSUFFICIENT_INFORMATION: The document does not describe quantum teleportation."

        result = generate_diagram(
            user_query="Quantum teleportation diagram",
            retrieved_context="This document covers only classical networking.",
            llm=mock_llm,
        )

        assert result["success"] is False
        assert "INSUFFICIENT_INFORMATION" in result["error"]
