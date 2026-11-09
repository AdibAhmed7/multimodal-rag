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
    parser = argparse.ArgumentParser(description="Test image retrieval using SigLIP 2.")
    parser.add_argument(
        "--query",
        default="Find diagrams explaining the main concept.",
        help="Text query to search for images",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Optional document ID to filter results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return",
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
        results = pipeline.search_images(
            args.query,
            limit=args.limit,
            document_id=args.document_id,
        )

        print(f"\nQuery: '{args.query}'")
        print(f"Retrieved {len(results)} images:")
        print("-" * 70)
        for rank, res in enumerate(results, start=1):
            print(f"[{rank}] Score: {res['score']:.4f} | Image ID: {res['id']} | Page: {res['page_number']}")
            print(f"    Path: {res['image_path']}")
            print(f"    Dimensions: {res.get('width')}x{res.get('height')}")
            if res.get("caption"):
                print(f"    Caption: {res['caption']}")
            if res.get("surrounding_text"):
                print(f"    Surrounding text: {res['surrounding_text'][:100]}...")
            print("-" * 70)
    finally:
        manager.close()


if __name__ == "__main__":
    main()

