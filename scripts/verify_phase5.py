import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vectorstore.qdrant_manager import QdrantManager
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline


def main():
    print("=" * 70)
    print("PHASE 5: COMPREHENSIVE END-TO-END RETRIEVAL VERIFICATION")
    print("=" * 70)

    manager = QdrantManager(database_path="data/processed/qdrant")
    try:
        pipeline = MultimodalRetrievalPipeline(manager.client)

        # -------------------------------------------------------------
        # TEST 1: Text-to-Text Search
        # -------------------------------------------------------------
        print("\n[TEST 1] Text-to-Text Retrieval (BGE-M3)...")
        query_text = "attention mechanism and encoder decoder"
        text_results = pipeline.search_text(query_text, limit=3)
        assert len(text_results) > 0, "Expected non-empty text results!"
        print(f"  ✓ Query: '{query_text}'")
        print(f"  ✓ Retrieved {len(text_results)} chunks")
        for r in text_results:
            assert "chunk_id" in r and "content" in r and "page_number" in r
            print(f"    - Chunk {r['id']} (Page {r['page_number']}, Score: {r['score']:.4f}): {r['content'][:60]}...")

        # -------------------------------------------------------------
        # TEST 2: Metadata Filtering by Document ID
        # -------------------------------------------------------------
        print("\n[TEST 2] Metadata Filtering by document_id...")
        res_matching = pipeline.search_text("Transformer", limit=3, document_id="example")
        assert len(res_matching) > 0, "Expected results for valid document_id='example'"
        print(f"  ✓ Filter document_id='example': {len(res_matching)} results found.")

        res_none = pipeline.search_text("Transformer", limit=3, document_id="non_existent_doc_id")
        assert len(res_none) == 0, "Expected 0 results for non-existent document_id"
        print(f"  ✓ Filter document_id='non_existent_doc_id': 0 results found (correct).")

        # -------------------------------------------------------------
        # TEST 3: Text-to-Image Cross-Modal Retrieval
        # -------------------------------------------------------------
        print("\n[TEST 3] Text-to-Image Retrieval (SigLIP 2)...")
        query_img = "diagram showing system pipeline and components"
        img_results = pipeline.search_images(query_img, limit=2)
        assert len(img_results) > 0, "Expected non-empty image results!"
        print(f"  ✓ Query: '{query_img}'")
        print(f"  ✓ Retrieved {len(img_results)} images")
        for r in img_results:
            assert "image_id" in r and "image_path" in r
            img_path = Path(r["image_path"])
            assert img_path.exists(), f"Image path does not exist on disk: {img_path}"
            print(f"    - Image {r['id']} (Page {r['page_number']}, Score: {r['score']:.4f}) | File exists: {img_path.name}")

        # -------------------------------------------------------------
        # TEST 4: Image-to-Image Similarity Retrieval
        # -------------------------------------------------------------
        print("\n[TEST 4] Image-to-Image Similarity Search...")
        sample_img_path = "data/processed/images/example/page_002_img_001.png"
        i2i_results = pipeline.search_by_image(sample_img_path, limit=2)
        assert len(i2i_results) > 0, "Expected non-empty image-to-image results!"
        top_match = i2i_results[0]
        print(f"  ✓ Query Image: {sample_img_path}")
        print(f"  ✓ Top Match ID: {top_match['id']} (Cosine Score: {top_match['score']:.4f})")
        assert top_match["score"] > 0.999, f"Exact image query should yield ~1.0 cosine similarity, got {top_match['score']}"
        assert top_match["id"] == "page_002_img_001", "Expected top match to be page_002_img_001"

        # -------------------------------------------------------------
        # TEST 5: Multimodal Retrieval with Reciprocal Rank Fusion
        # -------------------------------------------------------------
        print("\n[TEST 5] Multimodal Retrieval with Reciprocal Rank Fusion (RRF)...")
        mm_query = "Explain Transformer model architecture and show the architecture diagram"
        mm_res = pipeline.search_multimodal(mm_query, text_limit=3, image_limit=2)
        fused = mm_res["fused_results"]
        assert len(fused) > 0, "Expected non-empty fused results!"
        print(f"  ✓ Query: '{mm_query}'")
        print(f"  ✓ Total Fused Results: {len(fused)}")

        # Verify descending order of RRF scores
        rrf_scores = [item["rrf_score"] for item in fused]
        assert rrf_scores == sorted(rrf_scores, reverse=True), "RRF scores must be sorted in descending order!"

        # Verify presence of both modalities in fused output
        types_present = set(item["result_type"] for item in fused)
        print(f"  ✓ Result types represented in fused ranking: {types_present}")
        assert "text" in types_present, "Expected text results in fused output"
        assert "image" in types_present, "Expected image results in fused output"

        for rank, item in enumerate(fused, start=1):
            print(f"    [{rank}] [{item['result_type'].upper()}] RRF: {item['rrf_score']:.5f} | ID: {item['id']} | Page: {item['page_number']}")

        print("\n" + "=" * 70)
        print("✅ ALL PHASE 5 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        manager.close()


if __name__ == "__main__":
    main()

