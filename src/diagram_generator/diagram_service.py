"""Core service for AI-powered diagram generation from retrieved PDF contexts."""

import os
from typing import Any, Dict, List, Optional

from src.diagram_generator.diagram_prompts import DiagramPromptBuilder
from src.diagram_generator.mermaid_validator import (
    detect_diagram_type,
    extract_mermaid_code,
    sanitize_mermaid_labels,
    validate_mermaid_syntax,
)
from src.llm.cloud_client import CloudMultimodalClient


def generate_diagram(
    user_query: str,
    retrieved_context: Dict[str, Any],
    diagram_type: str = "auto",
    direction: str = "TD",
    llm: Optional[CloudMultimodalClient] = None,
    max_correction_attempts: int = 2,
) -> Dict[str, Any]:
    """Generates, validates, and refines a Mermaid.js diagram grounded in retrieved document evidence.

    Args:
        user_query: Natural language instruction or request for the diagram.
        retrieved_context: Structured dictionary containing 'pages' list with text chunks.
        diagram_type: Diagram family (flowchart, mindmap, sequenceDiagram, etc.) or 'auto'.
        direction: Flow direction ('TD' or 'LR') for flowcharts/architectures.
        llm: CloudMultimodalClient instance (initialized from .env if omitted).
        max_correction_attempts: Maximum repair loops if syntax validation fails.

    Returns:
        Structured response dictionary containing status, diagram_type, mermaid_code, and sources.
    """
    if llm is None:
        llm = CloudMultimodalClient(provider="gemini")

    if isinstance(retrieved_context, dict):
        pages = retrieved_context.get("pages", [])
    elif isinstance(retrieved_context, list):
        pages = retrieved_context
    elif isinstance(retrieved_context, str):
        pages = [{
            "document_id": "document",
            "page_number": 1,
            "text_chunks": [{"chunk_id": "c1", "content": retrieved_context}],
        }]
    else:
        pages = []

    # Extract source citations from context pages
    sources: List[Dict[str, Any]] = []
    for p in pages:
        doc_id = p.get("document_id")
        page_num = p.get("page_number")
        for chunk in p.get("text_chunks", []):
            sources.append({
                "type": "text",
                "id": chunk.get("chunk_id"),
                "document_id": doc_id,
                "page_number": page_num,
                "content": chunk.get("content", "")[:180],
                "score": chunk.get("score"),
            })

    # Step 1: Build generation prompt
    prompt = DiagramPromptBuilder.build_generation_prompt(
        user_query=user_query,
        pages_context=pages,
        diagram_type=diagram_type,
        direction=direction,
    )

    # Step 2: Query LLM
    try:
        raw_response = llm.generate(
            system_prompt=DiagramPromptBuilder.SYSTEM_PROMPT,
            user_text=prompt,
            max_tokens=2048,
            temperature=0.1,
        )
    except Exception as e:
        return {
            "success": False,
            "diagram_type": diagram_type,
            "mermaid_code": "",
            "error": f"LLM generation failed: {str(e)}",
            "sources": sources,
        }

    # Check if LLM signaled insufficient context
    lower_resp = raw_response.lower()
    if (
        ("does not contain enough information" in lower_resp or "insufficient information" in lower_resp or "insufficient_information" in lower_resp)
        and "```mermaid" not in raw_response
    ):
        return {
            "success": False,
            "diagram_type": diagram_type,
            "mermaid_code": "",
            "error": "INSUFFICIENT_INFORMATION: The uploaded document does not contain sufficient information to generate this diagram reliably.",
            "sources": sources,
            "repair_attempts": 0,
        }

    # Step 3: Extract and sanitize code
    mermaid_code = extract_mermaid_code(raw_response)
    mermaid_code = sanitize_mermaid_labels(mermaid_code)
    resolved_type = detect_diagram_type(mermaid_code) if diagram_type == "auto" else diagram_type

    # Step 4: Validate syntax with correction retry loop
    is_valid, validation_err = validate_mermaid_syntax(mermaid_code)
    attempts = 0

    while not is_valid and attempts < max_correction_attempts:
        attempts += 1
        print(f"[Diagram Service] Validation failed ({validation_err}). Repair attempt {attempts}/{max_correction_attempts}...")
        correction_prompt = DiagramPromptBuilder.build_correction_prompt(
            original_query=user_query,
            failed_mermaid_code=mermaid_code,
            error_message=validation_err or "Syntax error",
            pages_context=pages,
        )

        try:
            repaired_raw = llm.generate(
                system_prompt=DiagramPromptBuilder.SYSTEM_PROMPT,
                user_text=correction_prompt,
                max_tokens=2048,
                temperature=0.1,
            )
            repaired_code = extract_mermaid_code(repaired_raw)
            repaired_code = sanitize_mermaid_labels(repaired_code)
            is_valid, validation_err = validate_mermaid_syntax(repaired_code)
            if is_valid or len(repaired_code) > len(mermaid_code):
                mermaid_code = repaired_code
                resolved_type = detect_diagram_type(mermaid_code)
        except Exception as retry_err:
            print(f"[Diagram Service] Repair request failed: {retry_err}")
            break

    if not is_valid:
        return {
            "success": False,
            "diagram_type": resolved_type,
            "mermaid_code": mermaid_code,
            "error": f"Syntax validation failed after {attempts} correction attempts: {validation_err}",
            "sources": sources,
            "repair_attempts": attempts,
        }

    return {
        "success": True,
        "diagram_type": resolved_type,
        "mermaid_code": mermaid_code,
        "error": None,
        "sources": sources,
        "repair_attempts": attempts,
    }
