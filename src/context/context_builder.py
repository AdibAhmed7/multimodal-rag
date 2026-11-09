from typing import Any, Dict, List, Optional, Union
from src.context.context_organizer import ContextOrganizer
from src.context.image_describer import ImageDescriber


class ContextBuilder:
    """Consolidates retrieved text and image records into a structured prompt-ready context package."""

    def __init__(
        self,
        enable_image_descriptions: bool = True,
        max_images_to_describe: int = 5,
        describer_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        self.organizer = ContextOrganizer()
        self.enable_image_descriptions = enable_image_descriptions
        self.max_images_to_describe = max_images_to_describe
        self.describer_model_name = describer_model_name
        self.image_describer: Optional[ImageDescriber] = None

    def _get_image_describer(self) -> ImageDescriber:
        """Lazily initialize the image describer when required."""
        if self.image_describer is None:
            self.image_describer = ImageDescriber(model_name=self.describer_model_name)
        return self.image_describer

    def build(
        self,
        query: str,
        retrieval_results: Union[Dict[str, Any], List[Dict[str, Any]]],
        image_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convert retrieval outputs into an integrated, page-grouped multimodal context."""
        # Unpack results whether passed as unified pipeline dict or separate lists
        if isinstance(retrieval_results, dict):
            text_items = retrieval_results.get("text_results", [])
            img_items = retrieval_results.get("image_results", [])
        else:
            text_items = retrieval_results
            img_items = image_results or []

        pages = self.organizer.organize(text_items, img_items)

        described_count = 0
        describer = None
        if self.enable_image_descriptions and self.max_images_to_describe > 0:
            describer = self._get_image_describer()

        for page in pages:
            for img in page.get("images", []):
                if describer is not None and described_count < self.max_images_to_describe:
                    desc = describer.describe_image(
                        image_path=img.get("image_path", ""),
                        surrounding_text=img.get("surrounding_text"),
                    )
                    img["description"] = desc
                    described_count += 1
                else:
                    img["description"] = None

        statistics = {
            "text_chunks_retrieved": len(text_items),
            "images_retrieved": len(img_items),
            "images_described": described_count,
            "pages_with_evidence": len(pages),
        }

        return {
            "query": query,
            "pages": pages,
            "statistics": statistics,
        }

