import uuid
from typing import Any, Dict, List, Sequence
from qdrant_client import QdrantClient, models


class TextIndexer:
    """Indexes text chunk vectors and metadata into Qdrant."""

    def __init__(self, client: QdrantClient):
        self.client = client

    def index(
        self,
        collection_name: str,
        vectors: Sequence[Any],
        records: List[Dict[str, Any]],
        batch_size: int = 64,
    ) -> int:
        """Upsert text vectors with their metadata into the specified Qdrant collection."""
        if len(vectors) != len(records):
            raise ValueError("Number of text vectors and records must match.")

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
                        str(record["chunk_id"]),
                    )
                )
                vector_data = vector.tolist() if hasattr(vector, "tolist") else list(vector)

                payload = {
                    "chunk_id": record["chunk_id"],
                    "document_id": record.get("document_id"),
                    "page_number": record.get("page_number"),
                    "chunk_index": record.get("chunk_index"),
                    "content": record.get("content"),
                    "source_file": record.get("source_file"),
                    "related_image_ids": record.get("related_image_ids", []),
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

