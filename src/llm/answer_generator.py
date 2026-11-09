from pathlib import Path
from typing import Any, Dict, List, Optional
from src.llm.model_loader import QwenModel
from src.llm.prompt_builder import PromptBuilder


class AnswerGenerator:
    """Orchestrates prompt creation, image attachment, and Qwen multimodal generation."""

    def __init__(
        self,
        model: Optional[QwenModel] = None,
        max_images: int = 4,
        max_new_tokens: int = 700,
    ):
        self.model = model
        self.prompt_builder = PromptBuilder()
        self.max_images = max_images
        self.max_new_tokens = max_new_tokens

    def _get_model(self) -> QwenModel:
        """Lazily initialize the model if not injected."""
        if self.model is None:
            self.model = QwenModel()
        return self.model

    def generate(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send prompt evidence and attached raster figures to Qwen."""
        pages = context.get("pages", [])
        user_prompt = self.prompt_builder.build(query=query, pages=pages)

        content: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        attached_images: List[Dict[str, Any]] = []
        image_count = 0

        # Collect sources and attach actual image files
        for page in pages:
            doc_id = page.get("document_id")
            page_num = page.get("page_number")

            # Collect text sources
            for chunk in page.get("text_chunks", []):
                sources.append({
                    "type": "text",
                    "id": chunk.get("chunk_id"),
                    "document_id": doc_id,
                    "page_number": page_num,
                    "source_file": chunk.get("source_file"),
                    "content": chunk.get("content"),
                    "score": chunk.get("score"),
                })

            # Collect image sources and prepare image attachments
            for image in page.get("images", []):
                sources.append({
                    "type": "image",
                    "id": image.get("image_id"),
                    "document_id": doc_id,
                    "page_number": page_num,
                    "image_path": image.get("image_path"),
                    "description": image.get("description"),
                    "score": image.get("score"),
                })

                if image_count < self.max_images:
                    img_path_str = image.get("image_path")
                    if img_path_str:
                        path_obj = Path(img_path_str)
                        if not path_obj.exists():
                            print(f"Warning: Image not found: {path_obj}")
                        else:
                            content.append({
                                "type": "text",
                                "text": (
                                    f"Original PDF image: {image.get('image_id')}, "
                                    f"document: {doc_id}, page: {page_num}."
                                ),
                            })
                            content.append({
                                "type": "image",
                                "image": str(path_obj.resolve()),
                            })
                            attached_images.append({
                                "image_id": image.get("image_id"),
                                "document_id": doc_id,
                                "page_number": page_num,
                                "image_path": str(path_obj.resolve()),
                                "description": image.get("description"),
                            })
                            image_count += 1

        # Append textual prompt to content sequence
        content.append({
            "type": "text",
            "text": user_prompt,
        })

        messages = [
            {
                "role": "system",
                "content": self.prompt_builder.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        model = self._get_model()
        answer = model.generate(
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "attached_images": attached_images,
        }

