from collections import defaultdict
from typing import Any, Dict, List


class ContextOrganizer:
    """Groups retrieved text chunks and images by document and page while preserving relationships."""

    def organize(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Organize flat retrieval results into a hierarchical page-by-page structure."""
        documents: Dict[str, Dict[int, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
            lambda: defaultdict(lambda: {"text_chunks": [], "images": []})
        )

        # 1. Group text chunks
        for result in text_results:
            doc_id = result.get("document_id")
            page_num = result.get("page_number")
            if doc_id is None or page_num is None:
                continue

            # Copy result dictionary to avoid mutating input objects
            chunk_entry = {
                "chunk_id": result.get("chunk_id", result.get("id")),
                "chunk_index": result.get("chunk_index", 0),
                "content": result.get("content", ""),
                "source_file": result.get("source_file"),
                "score": result.get("score"),
                "related_image_ids": result.get("related_image_ids", []),
            }
            documents[doc_id][page_num]["text_chunks"].append(chunk_entry)

        # 2. Group images
        for result in image_results:
            doc_id = result.get("document_id")
            page_num = result.get("page_number")
            if doc_id is None or page_num is None:
                continue

            img_entry = {
                "image_id": result.get("image_id", result.get("id")),
                "image_index": result.get("image_index", 0),
                "image_path": result.get("image_path", ""),
                "width": result.get("width"),
                "height": result.get("height"),
                "caption": result.get("caption"),
                "surrounding_text": result.get("surrounding_text"),
                "score": result.get("score"),
                "related_text_chunk_ids": result.get("related_text_chunk_ids", []),
            }
            documents[doc_id][page_num]["images"].append(img_entry)

        # 3. Build ordered pages list
        pages: List[Dict[str, Any]] = []
        for doc_id in sorted(documents.keys()):
            for page_num in sorted(documents[doc_id].keys()):
                page_data = documents[doc_id][page_num]

                # Sort text chunks by original chunk_index
                page_data["text_chunks"].sort(
                    key=lambda c: c.get("chunk_index")
                    if c.get("chunk_index") is not None
                    else 0
                )

                # Sort images by image_index
                page_data["images"].sort(
                    key=lambda img: img.get("image_index")
                    if img.get("image_index") is not None
                    else 0
                )

                pages.append({
                    "document_id": doc_id,
                    "page_number": page_num,
                    "text_chunks": page_data["text_chunks"],
                    "images": page_data["images"],
                })

        return pages
