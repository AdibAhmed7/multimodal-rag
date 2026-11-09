import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vectorstore.qdrant_manager import QdrantManager
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline


def main():
    parser = argparse.ArgumentParser(description="Test multimodal retrieval with Reciprocal Rank Fusion.")
    parser.add_argument(
        "--query",
        default="Explain the main concept and show related diagrams.",
        help="Multimodal user query",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Optional document ID to filter results",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=5,
        help="Max text results to retrieve before fusion",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=5,
        help="Max image results to retrieve before fusion",
    )
    parser.add_argument(
        "--database",
        default="data/processed/qdrant",
        help="Path to local Qdrant database",
    )
    args = parser.parse_args()

    manager = QdrantManager(database_path=args.database)
    try:
        pipeline = MultimodalRetrievalPipeline(manager.client)
        output = pipeline.search_multimodal(
            args.query,
            text_limit=args.text_limit,
            image_limit=args.image_limit,
            document_id=args.document_id,
        )

        fused = output["fused_results"]
        print(f"\nQuery: '{args.query}'")
        print(f"Fused Multimodal Results ({len(fused)} items):")
        print("=" * 75)
        for rank, res in enumerate(fused, start=1):
            rtype = res.get("result_type", "unknown").upper()
            rrf = res.get("rrf_score", 0.0)
            orig_score = res.get("score", 0.0)
            item_id = res.get("id")
            page = res.get("page_number")

            print(f"[{rank}] [{rtype}] RRF Score: {rrf:.5f} (Raw Score: {orig_score:.4f}) | ID: {item_id} | Page: {page}")
            if rtype == "TEXT":
                print(f"    Snippet: {res.get('content', '')[:120]}...")
            elif rtype == "IMAGE":
                print(f"    Image Path: {res.get('image_path')}")
                if res.get("caption"):
                    print(f"    Caption: {res.get('caption')}")
            print("-" * 75)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
