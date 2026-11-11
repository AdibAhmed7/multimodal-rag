from pathlib import Path
from typing import Any, Dict, List, Optional
import os
from src.llm.cloud_client import CloudMultimodalClient
from src.llm.model_loader import QwenModel
from src.llm.prompt_builder import PromptBuilder


class AnswerGenerator:
    """Orchestrates prompt creation, image attachment, and cloud multimodal generation."""

    def __init__(
        self,
        model: Optional[QwenModel] = None,
        max_images: int = 4,
        max_new_tokens: int = 700,
        fast_mode: bool = False,
        online_mode: bool = True,
        cloud_provider: str = "gemini",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model = model
        self.prompt_builder = PromptBuilder()
        self.max_images = max_images
        self.max_new_tokens = max_new_tokens
        self.online_mode = online_mode
        self.cloud_provider = cloud_provider
        self.api_key = api_key

        self.cloud_client: Optional[CloudMultimodalClient] = None
        if self.model is None:
            try:
                self.cloud_client = CloudMultimodalClient(
                    provider=self.cloud_provider,
                    api_key=self.api_key,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize cloud multimodal provider '{self.cloud_provider}': {e}. "
                    f"Please ensure {self.cloud_provider.upper()}_API_KEY is set in your .env file."
                ) from e

    def generate(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send prompt evidence and attached raster figures to Cloud Multimodal Provider."""
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

        # Injected mock model (for unit testing)
        if self.model is not None:
            messages = [
                {"role": "system", "content": self.prompt_builder.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            answer = self.model.generate(
                messages=messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "attached_images": attached_images,
                "provider": "mock",
            }

        # Online Cloud Generation (Gemini / OpenAI)
        if not self.cloud_client:
            raise RuntimeError(
                f"Cloud client for '{self.cloud_provider}' is not initialized. "
                f"Please ensure {self.cloud_provider.upper()}_API_KEY is defined in your .env file."
            )

        img_paths = [img["image_path"] for img in attached_images if img.get("image_path")]
        print(f"[QA Engine] Dispatching to online cloud provider: '{self.cloud_provider.upper()}'...")
        answer = self.cloud_client.generate(
            system_prompt=self.prompt_builder.SYSTEM_PROMPT,
            user_text=user_prompt,
            image_paths=img_paths,
            max_tokens=self.max_new_tokens,
        )
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "attached_images": attached_images,
            "provider": f"online_{self.cloud_provider}",
        }

