"""Chat Backend Interface for Streamlit UI.

Communicates with the multimodal RAG orchestrator (ask_multimodal.py)
via isolated subprocess execution to prevent memory contention.
"""

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ChatInterface:
    """Invokes backend multimodal retrieval and generation scripts."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or PROJECT_ROOT

    def ask(
        self,
        query: str,
        document_id: Optional[str] = None,
        text_limit: int = 5,
        image_limit: int = 4,
        operation: str = "auto",
        cloud_provider: str = "gemini",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sends a query to the multimodal pipeline and loads the response payload."""
        resp_id = str(uuid.uuid4())
        out_file = self.root / "data" / "generated" / "responses" / f"resp_{resp_id}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(self.root / "scripts" / "ask_multimodal.py"),
            "--query",
            query,
            "--output",
            str(out_file),
            "--text-limit",
            str(text_limit),
            "--image-limit",
            str(image_limit),
            "--operation",
            operation,
            "--online",
            "--provider",
            cloud_provider,
        ]

        if document_id:
            command.extend(["--document-id", document_id])

        if api_key:
            command.extend(["--api-key", api_key])

        result = subprocess.run(
            command,
            cwd=str(self.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            raw_err = result.stderr.strip() or result.stdout.strip()
            # Clean up progress bar lines from error message
            clean_lines = [
                line for line in raw_err.splitlines()
                if "Loading weights:" not in line and not line.startswith("Loading weights")
            ]
            clean_err = "\n".join(clean_lines).strip() or "Process terminated unexpectedly."
            raise RuntimeError(f"Pipeline execution failed:\n{clean_err}")

        # Load generated response payload
        target_path = out_file
        if not target_path.exists():
            fallback_path = self.root / "data" / "generated" / "responses" / "latest_visual_response.json"
            if fallback_path.exists():
                target_path = fallback_path
            else:
                raise FileNotFoundError(f"Pipeline did not generate a response JSON at {out_file}")

        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)

