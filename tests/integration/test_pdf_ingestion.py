import json
from pathlib import Path
import pytest


@pytest.mark.integration
def test_pdf_ingestion():
    metadata_path = Path("data/processed/metadata/example.json")
    if not metadata_path.exists():
        metadata_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "metadata"
            / "example.json"
        )
    assert metadata_path.exists(), "Run Phase 1 PDF ingestion first."
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    assert metadata
    assert metadata.get("page_count", 0) > 0
    assert len(metadata.get("pages", [])) > 0
