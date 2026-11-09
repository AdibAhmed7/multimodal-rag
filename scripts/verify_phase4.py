import json
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vectorstore.qdrant_manager import QdrantManager
from src.vectorstore.index_text import TextIndexer
from src.vectorstore.index_images import ImageIndexer


def main():
    print("=== 1. Checking Database Files on Disk ===")
    db_dir = Path("data/processed/qdrant")
    assert db_dir.exists() and any(db_dir.iterdir()), "Qdrant database directory is empty!"
    print(f"Database directory: {db_dir.resolve()}")

    with open("data/processed/embeddings/example/manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    npz = np.load("data/processed/embeddings/example/vectors.npz")
    text_vectors = npz["text_vectors"]
    image_vectors = npz["image_vectors"]

    with QdrantManager("data/processed/qdrant") as manager:
        print("\n=== 2. Validating text_chunks Collection ===")
        info_text = manager.get_collection_info("text_chunks")
        print(f"Status: {info_text.status}")
        print(f"Points count: {info_text.points_count}")
        params_text = info_text.config.params.vectors
        print(f"Vector size: {params_text.size}, Distance: {params_text.distance}")
        assert info_text.points_count == len(manifest["text_records"]), "Text points count mismatch!"
        assert params_text.size == 1024, "Text vector dimension is not 1024!"

        records_text, _ = manager.client.scroll("text_chunks", limit=10, with_payload=True, with_vectors=True)
        for r in records_text:
            payload = r.payload
            print(f"  ✓ Point ID: {r.id} | Chunk: {payload['chunk_id']} | Page: {payload['page_number']} | Related Images: {payload['related_image_ids']}")
            assert len(r.vector) == 1024
            assert payload["content"]

        print("\n=== 3. Validating document_images Collection ===")
        info_img = manager.get_collection_info("document_images")
        print(f"Status: {info_img.status}")
        print(f"Points count: {info_img.points_count}")
        params_img = info_img.config.params.vectors
        print(f"Vector size: {params_img.size}, Distance: {params_img.distance}")
        assert info_img.points_count == len(manifest["image_records"]), "Image points count mismatch!"
        assert params_img.size == 768, "Image vector dimension is not 768!"

        records_img, _ = manager.client.scroll("document_images", limit=10, with_payload=True, with_vectors=True)
        for r in records_img:
            payload = r.payload
            img_file = Path(payload["image_path"])
            print(f"  ✓ Point ID: {r.id} | Img ID: {payload['image_id']} | Exists on disk: {img_file.exists()} | Dimensions: {payload['width']}x{payload['height']}")
            assert len(r.vector) == 768
            assert img_file.exists(), f"Image file does not exist: {img_file}"

        print("\n=== 4. Testing Cosine Similarity Query on text_chunks ===")
        query_vec = text_vectors[0].tolist()
        search_res = manager.client.query_points(collection_name="text_chunks", query=query_vec, limit=1)
        top_hit = search_res.points[0]
        print(f"Top match point ID: {top_hit.id} (Score: {top_hit.score:.4f})")
        print(f"Top match content:  {top_hit.payload['content'][:60]}...")
        assert top_hit.score > 0.999, "Cosine self-similarity should be ~1.0!"

        print("\n=== 5. Testing Cosine Similarity Query on document_images ===")
        query_img_vec = image_vectors[0].tolist()
        search_img_res = manager.client.query_points(collection_name="document_images", query=query_img_vec, limit=1)
        top_img_hit = search_img_res.points[0]
        print(f"Top match point ID: {top_img_hit.id} (Score: {top_img_hit.score:.4f})")
        print(f"Top match image ID: {top_img_hit.payload['image_id']}")
        assert top_img_hit.score > 0.999, "Cosine self-similarity should be ~1.0!"

        print("\n=== 6. Testing Idempotency (Re-indexing same data) ===")
        text_indexer = TextIndexer(manager.client)
        text_indexer.index("text_chunks", text_vectors, manifest["text_records"])
        image_indexer = ImageIndexer(manager.client)
        image_indexer.index("document_images", image_vectors, manifest["image_records"])

        assert manager.get_collection_info("text_chunks").points_count == len(manifest["text_records"])
        assert manager.get_collection_info("document_images").points_count == len(manifest["image_records"])
        print("Re-indexing completed without creating duplicate points (UUIDv5 idempotency confirmed).")

    print("\n✅ ALL PHASE 4 VALIDATIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
