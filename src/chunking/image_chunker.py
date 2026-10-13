from pathlib import Path
from src.chunking.chunk_models import ImageRecord, TextChunk


class ImageChunker:
    def chunk_document(
        self,
        document: dict,
        text_chunks: list[TextChunk],
    ) -> list[ImageRecord]:
        image_records = []
        document_id = document["document_id"]

        # Group text chunks by page so each image can reference
        # the text chunks from its own page.
        chunks_by_page: dict[int, list[TextChunk]] = {}
        for chunk in text_chunks:
            chunks_by_page.setdefault(chunk.page_number, []).append(chunk)

        for page in document["pages"]:
            page_number = page["page_number"]
            page_text_chunks = chunks_by_page.get(page_number, [])
            page_text = page.get("text", "").strip()

            for image_index, image in enumerate(page.get("images", []), start=1):
                image_id = image.get(
                    "image_id",
                    f"{document_id}_p{page_number:03d}_img{image_index:03d}",
                )
                image_path = image.get("path") or image.get("image_path", "")
                extension = image.get("extension", "")
                if not extension and image_path:
                    extension = Path(image_path).suffix.lstrip(".")

                related_text_chunk_ids = [
                    chunk.chunk_id for chunk in page_text_chunks
                ]

                # Link back to text chunks on the same page
                for chunk in page_text_chunks:
                    if image_id not in chunk.related_image_ids:
                        chunk.related_image_ids.append(image_id)

                image_records.append(
                    ImageRecord(
                        image_id=image_id,
                        document_id=document_id,
                        page_number=page_number,
                        image_index=image_index,
                        image_path=image_path,
                        extension=extension,
                        width=image.get("width"),
                        height=image.get("height"),
                        caption=image.get("caption"),
                        surrounding_text=page_text,
                        related_text_chunk_ids=related_text_chunk_ids,
                    )
                )

        return image_records
