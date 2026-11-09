from pathlib import Path
import pytest
from src.vectorstore.qdrant_manager import QdrantManager


@pytest.mark.integration
def test_qdrant_collections():
    repo_root = Path(__file__).resolve().parents[2]
    db_path = repo_root / "data" / "processed" / "qdrant"
    manager = QdrantManager(database_path=db_path)
    try:
        assert manager.client.collection_exists("text_chunks")
        assert manager.client.collection_exists("document_images")
        text_info = manager.get_collection_info("text_chunks")
        image_info = manager.get_collection_info("document_images")
        assert text_info.points_count > 0
        assert image_info.points_count > 0
    finally:
        manager.close()

