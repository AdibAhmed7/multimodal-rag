from src.chunking.chunk_models import TextChunk


class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: dict) -> list[TextChunk]:
        chunks = []
        document_id = document["document_id"]
        source_file = document["filename"]

        for page in document["pages"]:
            text = page.get("text", "").strip()
            if not text:
                continue

            page_number = page["page_number"]
            start = 0
            page_chunk_index = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))

                # Prefer a nearby paragraph or sentence boundary.
                if end < len(text):
                    boundary = max(
                        text.rfind("\n", start, end),
                        text.rfind(". ", start, end),
                        text.rfind(" ", start, end),
                    )
                    if boundary > start + self.chunk_size // 2:
                        end = boundary + 1

                content = text[start:end].strip()
                if content:
                    chunks.append(
                        TextChunk(
                            chunk_id=(
                                f"{document_id}_p{page_number:03d}"
                                f"_c{page_chunk_index:03d}"
                            ),
                            document_id=document_id,
                            page_number=page_number,
                            chunk_index=page_chunk_index,
                            content=content,
                            source_file=source_file,
                        )
                    )
                    page_chunk_index += 1

                if end >= len(text):
                    break

                start = max(start + 1, end - self.overlap)

        return chunks

    def chunk_text(
        self,
        text: str,
        document_id: str = "test_document",
        page_number: int = 1,
        source_file: str = "test.pdf",
    ) -> list[TextChunk]:
        """Convenience method to chunk raw text for a specific page."""
        return self.chunk_document({
            "document_id": document_id,
            "filename": source_file,
            "pages": [{"page_number": page_number, "text": text}],
        })

