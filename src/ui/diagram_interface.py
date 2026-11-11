"""Diagram Backend Interface for Streamlit UI.

Communicates with scripts/generate_diagram.py via isolated subprocess execution
to prevent memory contention and threading conflicts with Streamlit.
"""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DiagramInterface:
    """Invokes backend Mermaid diagram generation scripts."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or PROJECT_ROOT

    def generate(
        self,
        query: str,
        document_id: Optional[str] = None,
        diagram_type: str = "auto",
        direction: str = "TD",
        text_limit: int = 6,
        cloud_provider: str = "gemini",
    ) -> Dict[str, Any]:
        """Generates a grounded Mermaid diagram from the active document."""
        job_id = str(uuid.uuid4())
        out_file = self.root / "data" / "generated" / "diagrams" / f"diag_{job_id}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(self.root / "scripts" / "generate_diagram.py"),
            "--query",
            query,
            "--output",
            str(out_file),
            "--type",
            diagram_type,
            "--direction",
            direction,
            "--text-limit",
            str(text_limit),
            "--provider",
            cloud_provider,
        ]

        if document_id:
            command.extend(["--document-id", document_id])

        result = subprocess.run(
            command,
            cwd=str(self.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            raw_err = result.stderr.strip() or result.stdout.strip()
            clean_lines = [
                line for line in raw_err.splitlines()
                if "Loading weights:" not in line and not line.startswith("Loading weights")
            ]
            clean_err = "\n".join(clean_lines).strip() or "Diagram generation failed."
            raise RuntimeError(f"Diagram generator error:\n{clean_err}")

        if not out_file.exists():
            raise FileNotFoundError(f"Diagram result was not generated at {out_file}")

        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)

