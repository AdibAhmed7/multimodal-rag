from pathlib import Path
from typing import Optional
from src.ingestion.text_extractor import TextExtractor
from src.ingestion.image_extractor import ImageExtractor


class PageParser:
    """Parse pages into a unified representation containing both text and images."""

    def __init__(
        self,
        text_extractor: Optional[TextExtractor] = None,
        image_extractor: Optional[ImageExtractor] = None,
    ):
        self.text_extractor = text_extractor or TextExtractor()
        self.image_extractor = image_extractor or ImageExtractor()

    def parse_page(
        self,
        pdf_loader,
        page_number: int,
        image_output_dir: str | Path,
    ) -> dict:
        """Parse a single page (1-based index) and extract text and images."""
        page = pdf_loader.get_page(page_number - 1)
        text = self.text_extractor.extract_page(page)
        images = self.image_extractor.extract_page_images(
            page=page,
            page_number=page_number,
            output_dir=image_output_dir,
        )
        return {
            "page_number": page_number,
            "text": text,
            "images": images,
        }

    def parse_document(
        self,
        pdf_loader,
        image_output_dir: str | Path,
    ) -> list[dict]:
        """Parse all pages across the PDF document."""
        pages = []
        for page_number in range(1, pdf_loader.page_count + 1):
            page_dict = self.parse_page(
                pdf_loader=pdf_loader,
                page_number=page_number,
                image_output_dir=image_output_dir,
            )
            pages.append(page_dict)
        return pages
