from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient, models


class QdrantManager:
    """Manages local embedded Qdrant database instance and collection lifecycle."""

    def __init__(self, database_path: str | Path = "data/processed/qdrant"):
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.database_path))

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        """Create a collection if it does not already exist."""
        if self.client.collection_exists(collection_name):
            print(f"Collection already exists: {collection_name}")
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
        print(f"Created collection: {collection_name}")

    def get_collection_info(self, collection_name: str) -> models.CollectionInfo:
        """Retrieve detailed information about a collection."""
        return self.client.get_collection(collection_name=collection_name)

    def close(self) -> None:
        """Close the local embedded Qdrant client connection."""
        if hasattr(self.client, "close"):
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

