from datetime import datetime, timezone
from typing import Any, Dict


class ResponseFormatter:
    """Formats raw multimodal LLM outputs into structured API and presentation payloads."""

    @staticmethod
    def format(result: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate evidence sources, attached imagery, and answer text."""
        raw_sources = result.get("sources", [])
        image_sources = [
            source for source in raw_sources if source.get("type") == "image"
        ]
        text_sources = [
            source for source in raw_sources if source.get("type") == "text"
        ]

        unique_pages = sorted({
            (source.get("document_id"), source.get("page_number"))
            for source in raw_sources
            if source.get("document_id") is not None
            and source.get("page_number") is not None
        })

        attached_images = result.get("attached_images", [])

        return {
            "query": result.get("query", ""),
            "answer": result.get("answer", ""),
            "model": "Qwen2.5-VL-3B-Instruct",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "pages": [
                    {"document_id": doc, "page_number": page}
                    for doc, page in unique_pages
                ],
                "text_chunks": text_sources,
                "images": image_sources,
            },
            "display": {
                "images": attached_images,
            },
            "statistics": {
                "attached_image_count": len(attached_images),
                "text_source_count": len(text_sources),
                "image_source_count": len(image_sources),
            },
        }

