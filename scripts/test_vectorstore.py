import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vectorstore.qdrant_manager import QdrantManager


def main():
    parser = argparse.ArgumentParser(
        description="Verify and inspect collections in local Qdrant vectorstore."
    )
    parser.add_argument(
        "--database",
        default="data/processed/qdrant",
        help="Path to local Qdrant storage directory",
    )
    args = parser.parse_args()

    manager = QdrantManager(args.database)
    try:
        collections = ["text_chunks", "document_images"]
        for collection_name in collections:
            if not manager.client.collection_exists(collection_name):
                print(f"FAILED: {collection_name} does not exist.")
                continue

            info = manager.get_collection_info(collection_name)
            print(f"\nCollection: {collection_name}")
            print(f"Points: {info.points_count}")
            print(f"Status: {info.status}")

            records, _ = manager.client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if records:
                print(f"Sample Payload: {records[0].payload}")
            else:
                print("No points found in collection.")
    finally:
        manager.close()


if __name__ == "__main__":
    main()

