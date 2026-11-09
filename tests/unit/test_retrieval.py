import numpy as np
import pytest
from unittest.mock import MagicMock
from src.retrieval.result_fusion import reciprocal_rank_fusion
from src.retrieval.text_retriever import TextRetriever
from src.retrieval.image_retriever import ImageRetriever
from src.vectorstore.qdrant_manager import QdrantManager
from src.vectorstore.index_text import TextIndexer
from src.vectorstore.index_images import ImageIndexer


def test_reciprocal_rank_fusion_basic():
    text_results = [
        {"id": "c1", "result_type": "text", "score": 0.9},
        {"id": "c2", "result_type": "text", "score": 0.8},
    ]
    image_results = [
        {"id": "img1", "result_type": "image", "score": 0.85},
        {"id": "img2", "result_type": "image", "score": 0.75},
    ]

    fused = reciprocal_rank_fusion([text_results, image_results], k=60)
    assert len(fused) == 4
    # First rank in each list has 1 / (60 + 1) = 1/61
    assert pytest.approx(fused[0]["rrf_score"]) == 1.0 / 61
    assert pytest.approx(fused[1]["rrf_score"]) == 1.0 / 61
    # Second rank in each list has 1 / (60 + 2) = 1/62
    assert pytest.approx(fused[2]["rrf_score"]) == 1.0 / 62
    assert pytest.approx(fused[3]["rrf_score"]) == 1.0 / 62


def test_reciprocal_rank_fusion_empty():
    fused = reciprocal_rank_fusion([])
    assert fused == []

    fused = reciprocal_rank_fusion([[], []])
    assert fused == []


def test_text_retriever_search(tmp_path):
    db_path = tmp_path / "qdrant"
    with QdrantManager(database_path=db_path) as manager:
        manager.create_collection("text_chunks", vector_size=4)
        indexer = TextIndexer(manager.client)

        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        records = [
            {"chunk_id": "c1", "document_id": "doc_a", "page_number": 1, "content": "Text A"},
            {"chunk_id": "c2", "document_id": "doc_b", "page_number": 2, "content": "Text B"},
        ]
        indexer.index("text_chunks", vectors, records)

        retriever = TextRetriever(manager.client, collection_name="text_chunks")

        # Query close to c1
        results = retriever.search([1.0, 0.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0]["id"] == "c1"
        assert results[0]["result_type"] == "text"
        assert results[0]["content"] == "Text A"
        assert pytest.approx(results[0]["score"], rel=1e-2) == 1.0

        # Query with document_id filter
        filtered_results = retriever.search([1.0, 0.0, 0.0, 0.0], limit=2, document_id="doc_b")
        assert len(filtered_results) == 1
        assert filtered_results[0]["id"] == "c2"
        assert filtered_results[0]["document_id"] == "doc_b"


def test_image_retriever_search(tmp_path):
    db_path = tmp_path / "qdrant"
    with QdrantManager(database_path=db_path) as manager:
        manager.create_collection("document_images", vector_size=3)
        indexer = ImageIndexer(manager.client)

        vectors = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        records = [
            {"image_id": "img1", "document_id": "doc_a", "page_number": 1, "image_path": "img1.png"},
            {"image_id": "img2", "document_id": "doc_b", "page_number": 2, "image_path": "img2.png"},
        ]
        indexer.index("document_images", vectors, records)

        mock_encoder = MagicMock()
        mock_encoder.encode_text_for_image_search.return_value = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        mock_encoder.encode_image.return_value = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        retriever = ImageRetriever(manager.client, query_encoder=mock_encoder, collection_name="document_images")

        # Search by text
        text_matches = retriever.search_by_text("find image 1", limit=1)
        assert len(text_matches) == 1
        assert text_matches[0]["id"] == "img1"
        assert text_matches[0]["result_type"] == "image"
        assert text_matches[0]["image_path"] == "img1.png"

        # Search by image
        img_matches = retriever.search_by_image("dummy_path", limit=1)
        assert len(img_matches) == 1
        assert img_matches[0]["id"] == "img2"
