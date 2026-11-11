"""Response builder for multimodal RAG interactions.

Merges generated answers, retrieved PDF images, and newly generated AI images
while keeping retrieved vs generated visual assets strictly separated.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class VisualResponseBuilder:
    """Builds a standardized visual response payload combining text and imagery."""

    def build(
        self,
        query: str,
        intent: str,
        answer: Optional[str] = None,
        retrieved_images: Optional[List[Dict[str, Any]]] = None,
        generated_images: Optional[List[Dict[str, Any]]] = None,
        diagram: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Assembles the final multimodal response payload.

        Args:
            query: The user's original query.
            intent: The resolved routing intent.
            answer: Generated text answer from the LLM, or None.
            retrieved_images: List of images retrieved from the PDF via Qdrant.
            generated_images: List of newly generated AI images.
            diagram: Structured Mermaid.js diagram payload (code, type, metadata).
            sources: Source references and citations.

        Returns:
            Structured dictionary with distinct fields for retrieved and generated imagery.
        """
        retrieved_images = retrieved_images or []
        generated_images = generated_images or []
        sources = sources or []

        return {
            "query": query,
            "intent": intent,
            "answer": answer,
            "retrieved_images": retrieved_images,
            "generated_images": generated_images,
            "diagram": diagram,
            "sources": sources,
            "statistics": {
                "retrieved_image_count": len(retrieved_images),
                "generated_image_count": len(generated_images),
                "has_diagram": bool(diagram and diagram.get("mermaid_code")),
                "source_count": len(sources),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

