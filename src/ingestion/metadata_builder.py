import json
from pathlib import Path


class MetadataBuilder:
    """Build and persist structured metadata for ingested documents."""

    def build(
        self,
        pdf_loader,
        pages: list[dict],
    ) -> dict:
        """Construct a standardized document metadata structure."""
        pdf_path = Path(pdf_loader.pdf_path)
        return {
            "document_id": pdf_path.stem,
            "filename": pdf_path.name,
            "source_path": str(pdf_path.as_posix()),
            "page_count": pdf_loader.page_count,
            "metadata": pdf_loader.metadata,
            "pages": pages,
        }

    def save(
        self,
        metadata: dict,
        output_path: str | Path,
    ):
        """Save document metadata JSON to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
