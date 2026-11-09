from typing import Any, Dict, List, Optional, Sequence, Union
from pathlib import Path
from PIL import Image
from qdrant_client import QdrantClient, models


class ImageRetriever:
    """Searches the document_images collection using SigLIP 2 query vectors."""

    def __init__(
        self,
        client: QdrantClient,
        query_encoder: Any,
        collection_name: str = "document_images",
    ):
        self.client = client
        self.query_encoder = query_encoder
        self.collection_name = collection_name

    def _search(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute vector search on the image collection."""
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        vector_data = (
            query_vector.tolist()
            if hasattr(query_vector, "tolist")
            else list(query_vector)
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector_data,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append({
                "id": payload.get("image_id", str(point.id)),
                "point_id": str(point.id),
                "score": float(point.score),
                "result_type": "image",
                "image_id": payload.get("image_id"),
                "image_path": payload.get("image_path"),
                "document_id": payload.get("document_id"),
                "page_number": payload.get("page_number"),
                "image_index": payload.get("image_index"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "caption": payload.get("caption"),
                "surrounding_text": payload.get("surrounding_text"),
                "related_text_chunk_ids": payload.get("related_text_chunk_ids", []),
            })
        return results

    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Text-to-image search using SigLIP 2 text embeddings."""
        query_vector = self.query_encoder.encode_text_for_image_search(query)
        return self._search(query_vector, limit=limit, document_id=document_id)

    def search_by_image(
        self,
        image_input: Union[str, Path, Image.Image],
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Image-to-image search using SigLIP 2 vision embeddings."""
        query_vector = self.query_encoder.encode_image(image_input)
        return self._search(query_vector, limit=limit, document_id=document_id)
