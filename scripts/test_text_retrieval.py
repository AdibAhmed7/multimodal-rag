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
    parser = argparse.ArgumentParser(description="Test text chunk retrieval using BGE-M3.")
    parser.add_argument(
        "--query",
        default="What is the main topic of this document?",
        help="Text query to search for",
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
        results = pipeline.search_text(
            args.query,
            limit=args.limit,
            document_id=args.document_id,
        )

        print(f"\nQuery: '{args.query}'")
        print(f"Retrieved {len(results)} text chunks:")
        print("-" * 70)
        for rank, res in enumerate(results, start=1):
            print(f"[{rank}] Score: {res['score']:.4f} | Chunk ID: {res['id']} | Page: {res['page_number']}")
            print(f"    Content: {res['content'][:120]}...")
            if res.get("related_image_ids"):
                print(f"    Related Images: {res['related_image_ids']}")
            print("-" * 70)
    finally:
        manager.close()


if __name__ == "__main__":
    main()

