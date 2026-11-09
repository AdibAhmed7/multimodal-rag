import uuid
import pytest
import numpy as np
from src.vectorstore.qdrant_manager import QdrantManager
from src.vectorstore.index_text import TextIndexer
from src.vectorstore.index_images import ImageIndexer


def test_qdrant_manager_create_and_info(tmp_path):
    db_path = tmp_path / "qdrant"
    with QdrantManager(database_path=db_path) as manager:
        manager.create_collection("test_collection", vector_size=128)
        assert manager.client.collection_exists("test_collection")

        # Creating again should be safe (no error)
        manager.create_collection("test_collection", vector_size=128)

        info = manager.get_collection_info("test_collection")
        assert info.points_count == 0


def test_text_indexer(tmp_path):
    db_path = tmp_path / "qdrant"
    with QdrantManager(database_path=db_path) as manager:
        manager.create_collection("text_chunks", vector_size=4)
        indexer = TextIndexer(manager.client)

        vectors = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ], dtype=np.float32)

        records = [
            {
                "chunk_id": "doc_p1_c0",
                "document_id": "doc",
                "page_number": 1,
                "chunk_index": 0,
                "content": "First chunk test content.",
                "source_file": "doc.pdf",
                "related_image_ids": ["img_01"],
            },
            {
                "chunk_id": "doc_p1_c1",
                "document_id": "doc",
                "page_number": 1,
                "chunk_index": 1,
                "content": "Second chunk test content.",
                "source_file": "doc.pdf",
                "related_image_ids": [],
            },
        ]

        count = indexer.index("text_chunks", vectors, records, batch_size=1)
        assert count == 2

        info = manager.get_collection_info("text_chunks")
        assert info.points_count == 2

        # Check payload and deterministic UUID
        expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "doc_p1_c0"))
        point = manager.client.retrieve("text_chunks", ids=[expected_uuid])
        assert len(point) == 1
        assert point[0].payload["chunk_id"] == "doc_p1_c0"
        assert point[0].payload["content"] == "First chunk test content."
        assert point[0].payload["related_image_ids"] == ["img_01"]

        # Re-indexing the same items should not duplicate points
        indexer.index("text_chunks", vectors, records)
        info_after = manager.get_collection_info("text_chunks")
        assert info_after.points_count == 2


def test_image_indexer(tmp_path):
    db_path = tmp_path / "qdrant"
    with QdrantManager(database_path=db_path) as manager:
        manager.create_collection("document_images", vector_size=3)
        indexer = ImageIndexer(manager.client)

        vectors = np.array([
            [0.1, 0.2, 0.3],
        ], dtype=np.float32)

        records = [
            {
                "image_id": "img_p2_01",
                "document_id": "doc",
                "page_number": 2,
                "image_index": 1,
                "image_path": "data/processed/images/doc/img.png",
                "extension": "png",
                "width": 800,
                "height": 600,
                "caption": "Test diagram",
                "surrounding_text": "Figure 1 diagram",
                "related_text_chunk_ids": ["doc_p2_c0"],
            }
        ]

        count = indexer.index("document_images", vectors, records)
        assert count == 1

        info = manager.get_collection_info("document_images")
        assert info.points_count == 1

        expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "img_p2_01"))
        point = manager.client.retrieve("document_images", ids=[expected_uuid])
        assert len(point) == 1
        assert point[0].payload["image_id"] == "img_p2_01"
        assert point[0].payload["width"] == 800
        assert point[0].payload["caption"] == "Test diagram"


def test_indexer_mismatch_raises(tmp_path):
    db_path = tmp_path / "qdrant"
    with QdrantManager(database_path=db_path) as manager:
        manager.create_collection("test_col", vector_size=2)
        indexer = TextIndexer(manager.client)

        vectors = np.array([[0.1, 0.2]])
        records = []

        with pytest.raises(ValueError, match="Number of text vectors and records must match."):
            indexer.index("test_col", vectors, records)

