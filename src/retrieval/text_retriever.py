from typing import Any, Dict, List, Optional, Sequence
from qdrant_client import QdrantClient, models


class TextRetriever:
    """Searches the text_chunks collection using BGE-M3 text query vectors."""

    def __init__(self, client: QdrantClient, collection_name: str = "text_chunks"):
        self.client = client
        self.collection_name = collection_name

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search text_chunks for the top matching chunks."""
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
                "id": payload.get("chunk_id", str(point.id)),
                "point_id": str(point.id),
                "score": float(point.score),
                "result_type": "text",
                "chunk_id": payload.get("chunk_id"),
                "content": payload.get("content", ""),
                "document_id": payload.get("document_id"),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "source_file": payload.get("source_file"),
                "related_image_ids": payload.get("related_image_ids", []),
            })
        return results

