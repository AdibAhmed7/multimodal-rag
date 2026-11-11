"""Validation and sanitization layer for Mermaid.js diagram syntax."""

import re
from typing import Optional, Tuple

SUPPORTED_DECLARATIONS = [
    "flowchart",
    "graph",
    "sequencediagram",
    "mindmap",
    "classdiagram",
    "erdiagram",
    "statediagram",
    "statediagram-v2",
    "timeline",
    "gantt",
    "gitgraph",
    "architecture-beta",
    "journey",
    "quadrantchart",
]

DISALLOWED_PATTERNS = [
    re.compile(r"<\s*script", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"on(load|error|click|mouseover)\s*=", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
]


def extract_mermaid_code(raw_text: str) -> str:
    """Extracts Mermaid code from fenced blocks or raw output."""
    if not raw_text:
        return ""

    # Check for ```mermaid ... ``` code fence
    fence_pattern = re.compile(r"```(?:mermaid)?\s*([\s\S]*?)```", re.IGNORECASE)
    match = fence_pattern.search(raw_text)
    if match:
        code = match.group(1).strip()
    else:
        # Fallback: find first line that starts with a known Mermaid declaration
        lines = [line.rstrip() for line in raw_text.splitlines() if line.strip()]
        start_idx = -1
        for idx, line in enumerate(lines):
            first_word = line.strip().split()[0].lower() if line.strip().split() else ""
            if any(first_word.startswith(decl) for decl in SUPPORTED_DECLARATIONS):
                start_idx = idx
                break

        if start_idx != -1:
            code = "\n".join(lines[start_idx:]).strip()
        else:
            code = raw_text.strip()

    # Clean any lingering code fence markers
    code = re.sub(r"^```mermaid\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"^```\s*", "", code)
    code = re.sub(r"\s*```$", "", code)

    return code.strip()


def sanitize_mermaid_labels(code: str) -> str:
    """Auto-repairs common Mermaid syntax flaws like unquoted parentheses in node brackets."""
    lines = []
    for line in code.splitlines():
        # Repair unquoted node labels: id[Text (Note)] -> id["Text (Note)"]
        def quote_bracket_contents(match):
            prefix = match.group(1)
            content = match.group(2)
            if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
                return f"{prefix}[{content}]"
            # If content has parentheses, brackets, colons, or slashes, wrap in quotes
            if any(ch in content for ch in "():/-#@,"):
                cleaned_content = content.replace('"', "'")
                return f'{prefix}["{cleaned_content}"]'
            return f"{prefix}[{content}]"

        repaired_line = re.sub(r'([a-zA-Z0-9_]+)\[([^\]\n]+)\]', quote_bracket_contents, line)
        lines.append(repaired_line)
    return "\n".join(lines)


def detect_diagram_type(code: str) -> str:
    """Determines the diagram type from the Mermaid declaration header."""
    cleaned = code.strip().lower()
    for decl in SUPPORTED_DECLARATIONS:
        if cleaned.startswith(decl):
            return decl
    return "diagram"


def validate_mermaid_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validates the syntax and security of extracted Mermaid code.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Empty Mermaid code generated."

    # Check for security hazards (XSS / script injection)
    for pattern in DISALLOWED_PATTERNS:
        if pattern.search(code):
            return False, "Disallowed script or executable pattern detected in diagram code."

    # Verify declaration header
    lines = [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith("%%")]
    if not lines:
        return False, "Diagram contains no executable Mermaid declarations."

    first_line = lines[0].lower()
    first_token = first_line.split()[0] if first_line.split() else ""

    valid_header = any(first_token.startswith(decl) for decl in SUPPORTED_DECLARATIONS)
    if not valid_header:
        return False, (
            f"Invalid Mermaid declaration '{first_token}'. "
            f"Must begin with one of: {', '.join(SUPPORTED_DECLARATIONS[:8])}..."
        )

    # Check balanced brackets and quotes
    bracket_pairs = [("[", "]"), ("{", "}"), ("(", ")")]
    for open_b, close_b in bracket_pairs:
        if code.count(open_b) != code.count(close_b):
            # For mindmap or certain diagrams this may vary, but for flowcharts it indicates broken syntax
            if first_token.startswith("flowchart") or first_token.startswith("graph"):
                return False, f"Unbalanced bracket delimiters: '{open_b}' and '{close_b}'."

    return True, None
