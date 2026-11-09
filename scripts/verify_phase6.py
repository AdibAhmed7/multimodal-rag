import json
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vectorstore.qdrant_manager import QdrantManager
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline
from src.context.context_organizer import ContextOrganizer
from src.context.context_builder import ContextBuilder
from src.context.image_describer import ImageDescriber


def test_existing_context_file():
    print("\n[CHECK 1] Validating latest_context.json Schema and Field Integrity...")
    context_file = Path("data/processed/contexts/latest_context.json")
    assert context_file.exists(), f"Context file does not exist: {context_file}"

    with open(context_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "query" in data and isinstance(data["query"], str)
    assert "pages" in data and isinstance(data["pages"], list)
    assert "statistics" in data and isinstance(data["statistics"], dict)

    stats = data["statistics"]
    for k in ["text_chunks_retrieved", "images_retrieved", "images_described", "pages_with_evidence"]:
        assert k in stats and isinstance(stats[k], int)

    assert len(data["pages"]) == stats["pages_with_evidence"]

    total_chunks = 0
    total_images = 0
    total_described = 0

    for p_idx, page in enumerate(data["pages"]):
        assert "document_id" in page and page["document_id"]
        assert "page_number" in page and isinstance(page["page_number"], int)
        assert "text_chunks" in page and isinstance(page["text_chunks"], list)
        assert "images" in page and isinstance(page["images"], list)

        # Check chunk sorting and fields
        prev_idx = -1
        for c in page["text_chunks"]:
            total_chunks += 1
            assert "chunk_id" in c and c["chunk_id"]
            assert "content" in c and c["content"]
            assert "source_file" in c and c["source_file"]
            curr_idx = c.get("chunk_index", 0)
            assert curr_idx >= prev_idx, f"Chunks not sorted by chunk_index on page {page['page_number']}"
            prev_idx = curr_idx

        # Check image fields & path validity
        for img in page["images"]:
            total_images += 1
            assert "image_id" in img and img["image_id"]
            assert "image_path" in img and img["image_path"]
            img_path = Path(img["image_path"])
            assert img_path.exists(), f"Image path in context does not exist: {img_path}"
            assert img.get("width") and img.get("height"), "Image dimensions missing"
            if img.get("description") is not None:
                total_described += 1
                assert len(img["description"]) > 10, "Description appears truncated"

    assert total_chunks == stats["text_chunks_retrieved"], "Text chunk count mismatch in stats"
    assert total_images == stats["images_retrieved"], "Image count mismatch in stats"
    assert total_described == stats["images_described"], "Described image count mismatch in stats"
    print(f"  ✓ Schema is valid: {total_chunks} chunks, {total_images} images, {total_described} described across {len(data['pages'])} pages.")


def test_edge_cases():
    print("\n[CHECK 2] Testing Edge Cases and Error Boundaries...")

    organizer = ContextOrganizer()

    # 1. Empty input
    empty_pages = organizer.organize([], [])
    assert empty_pages == [], "Empty input should produce empty pages"
    print("  ✓ Empty input handled cleanly.")

    # 2. Missing document_id / page_number
    malformed_records = [
        {"content": "No doc id"},
        {"document_id": "doc1", "content": "No page number"},
        {"page_number": 1, "content": "No doc id"},
    ]
    assert organizer.organize(malformed_records, []) == [], "Malformed records without doc/page should be safely skipped"
    print("  ✓ Missing document_id / page_number records safely skipped.")

    # 3. None chunk_index handled without crash
    none_index_records = [
        {"chunk_id": "c1", "document_id": "doc", "page_number": 1, "chunk_index": None, "content": "A"},
        {"chunk_id": "c2", "document_id": "doc", "page_number": 1, "chunk_index": 0, "content": "B"},
    ]
    pages = organizer.organize(none_index_records, [])
    assert len(pages) == 1
    assert len(pages[0]["text_chunks"]) == 2
    print("  ✓ None chunk_index handled without error.")

    # 4. ContextBuilder with max_images_to_describe limit
    builder = ContextBuilder(enable_image_descriptions=False, max_images_to_describe=0)
    ctx = builder.build("test", {"text_results": [], "image_results": []})
    assert ctx["statistics"]["images_described"] == 0
    assert ctx["statistics"]["pages_with_evidence"] == 0
    print("  ✓ Zero result ContextBuilder outputs valid structure.")


def test_cache_resilience(tmp_path):
    print("\n[CHECK 3] Testing ImageDescriber Cache Resilience...")

    # Corrupted cache file
    bad_cache = tmp_path / "corrupt_cache.json"
    bad_cache.write_text("{ this is not valid json }")
    describer = ImageDescriber(cache_path=str(bad_cache))
    assert describer.cache == {}, "Corrupted cache should reset to empty dict gracefully"

    # Non-existent image file
    missing_desc = describer.describe_image(tmp_path / "non_existent_image.png")
    assert "[Image file not found" in missing_desc
    print("  ✓ Cache corruption and non-existent image paths handled gracefully.")


def test_cli_variations():
    print("\n[CHECK 4] Testing Live Pipeline Retrieval with Constraints...")
    manager = QdrantManager(database_path="data/processed/qdrant")
    try:
        pipeline = MultimodalRetrievalPipeline(manager.client)

        # 1. Single chunk retrieval
        builder = ContextBuilder(enable_image_descriptions=True, max_images_to_describe=1)
        res = pipeline.search_multimodal("Transformer attention", text_limit=1, image_limit=1)
        ctx = builder.build("Transformer attention", res)

        assert ctx["statistics"]["text_chunks_retrieved"] == 1
        assert ctx["statistics"]["images_retrieved"] == 1
        assert ctx["statistics"]["images_described"] <= 1
        print("  ✓ Limit constrained multimodal retrieval and context build succeeded.")

        # 2. Document-filtered retrieval
        res_doc = pipeline.search_multimodal("Transformer", text_limit=3, image_limit=2, document_id="example")
        ctx_doc = builder.build("Transformer", res_doc)
        for page in ctx_doc["pages"]:
            assert page["document_id"] == "example"
        print("  ✓ Document-filtered context construction verified.")

    finally:
        manager.close()


def main():
    print("=" * 70)
    print("PHASE 6: COMPREHENSIVE PROBLEM & ERROR DIAGNOSTIC TEST")
    print("=" * 70)

    test_existing_context_file()
    test_edge_cases()
    test_cache_resilience(Path("data/processed"))
    test_cli_variations()

    print("\n" + "=" * 70)
    print("✅ ZERO PROBLEMS FOUND: PHASE 6 IS ROBUST AND FULLY OPERATIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    main()
