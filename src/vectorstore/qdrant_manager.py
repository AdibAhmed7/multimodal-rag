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
        max_retries: int = 5,
    ) -> None:
        """Create a collection if it does not already exist."""
        if self.client.collection_exists(collection_name):
            print(f"Collection already exists: {collection_name}")
            return

        import time
        import shutil

        for attempt in range(max_retries):
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                print(f"Created collection: {collection_name}")
                return
            except PermissionError as e:
                # Windows file locking can temporarily block os.replace on meta.json
                time.sleep(0.4)
                tmp_meta = self.database_path / "meta.json.tmp"
                target_meta = self.database_path / "meta.json"
                if tmp_meta.exists():
                    try:
                        shutil.copyfile(tmp_meta, target_meta)
                        tmp_meta.unlink(missing_ok=True)
                        print(f"Recovered and created collection: {collection_name}")
                        return
                    except Exception:
                        pass
                if attempt == max_retries - 1:
                    raise e

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

