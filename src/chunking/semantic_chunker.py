from src.chunking.text_chunker import TextChunker
from src.chunking.image_chunker import ImageChunker


class SemanticChunker:
    def __init__(
        self,
        text_chunker: TextChunker | None = None,
        image_chunker: ImageChunker | None = None,
    ):
        self.text_chunker = text_chunker or TextChunker()
        self.image_chunker = image_chunker or ImageChunker()

    def process_document(self, document: dict) -> dict:
        text_chunks = self.text_chunker.chunk_document(document)
        image_records = self.image_chunker.chunk_document(
            document=document,
            text_chunks=text_chunks,
        )
        return {
            "document_id": document["document_id"],
            "filename": document["filename"],
            "page_count": document["page_count"],
            "text_chunks": [
                chunk.model_dump() for chunk in text_chunks
            ],
            "image_records": [
                record.model_dump() for record in image_records
            ],
        }
