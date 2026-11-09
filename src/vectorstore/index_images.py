import uuid
from typing import Any, Dict, List, Sequence
from qdrant_client import QdrantClient, models


class ImageIndexer:
    """Indexes image vectors and metadata into Qdrant."""

    def __init__(self, client: QdrantClient):
        self.client = client

    def index(
        self,
        collection_name: str,
        vectors: Sequence[Any],
        records: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        """Upsert image vectors with their metadata into the specified Qdrant collection."""
        if len(vectors) != len(records):
            raise ValueError("Number of image vectors and records must match.")

        total = len(records)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_vectors = vectors[start:end]
            batch_records = records[start:end]

            points = []
            for vector, record in zip(batch_vectors, batch_records):
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        str(record["image_id"]),
                    )
                )
                vector_data = vector.tolist() if hasattr(vector, "tolist") else list(vector)

                payload = {
                    "image_id": record["image_id"],
                    "document_id": record.get("document_id"),
                    "page_number": record.get("page_number"),
                    "image_index": record.get("image_index"),
                    "image_path": record.get("image_path"),
                    "extension": record.get("extension"),
                    "width": record.get("width"),
                    "height": record.get("height"),
                    "caption": record.get("caption"),
                    "surrounding_text": record.get("surrounding_text"),
                    "related_text_chunk_ids": record.get("related_text_chunk_ids", []),
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector_data,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )

        return total

