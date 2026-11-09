from pathlib import Path
import pytest
from src.vectorstore.qdrant_manager import QdrantManager
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline


@pytest.mark.integration
def test_multimodal_retrieval():
    repo_root = Path(__file__).resolve().parents[2]
    db_path = repo_root / "data" / "processed" / "qdrant"
    manager = QdrantManager(database_path=db_path)
    try:
        pipeline = MultimodalRetrievalPipeline(manager.client)
        results = pipeline.search_multimodal(
            query="Explain the main concept of the document",
            text_limit=5,
            image_limit=5,
        )
        assert "text_results" in results
        assert "image_results" in results
        assert "combined_results" in results
        for item in results["text_results"]:
            assert item.get("content") is not None
    finally:
        manager.close()

