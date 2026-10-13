from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is not installed in the active Python environment. "
            "Please install it by running: pip install PyMuPDF"
        ) from exc


class PDFLoader:
    """Load a PDF document and expose basic document information."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {self.pdf_path.suffix}")
        self.document = fitz.open(self.pdf_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the PDF document."""
        if hasattr(self, "document") and self.document is not None:
            self.document.close()

    @property
    def page_count(self) -> int:
        return len(self.document)

    @property
    def metadata(self) -> dict:
        return self.document.metadata

    def get_page(self, page_number: int):
        """Return a zero-indexed PyMuPDF page."""
        if page_number < 0 or page_number >= self.page_count:
            raise IndexError(
                f"Page {page_number} is out of range (0 to {self.page_count - 1})."
            )
        return self.document[page_number]
