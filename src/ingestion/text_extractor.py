class TextExtractor:
    """Extract text from PDF pages."""

    def extract_page(self, page) -> str:
        """Extract plain text from a single page."""
        text = page.get_text("text")
        return text.strip()

    def extract_document(self, pdf_loader) -> list[dict]:
        """Extract text from all pages in a document using 1-based indexing."""
        pages = []
        for page_number in range(pdf_loader.page_count):
            page = pdf_loader.get_page(page_number)
            text = self.extract_page(page)
            pages.append({
                "page_number": page_number + 1,
                "text": text
            })
        return pages
