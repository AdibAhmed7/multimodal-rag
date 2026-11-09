import argparse
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


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Index Phase 3 embeddings and metadata into Qdrant."
    )
    parser.add_argument(
        "embedding_dir",
        help="Path to the Phase 3 embedding directory"
    )
    parser.add_argument(
        "--database",
        default="data/processed/qdrant",
        help="Path to the local Qdrant database directory"
    )
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    vectors_path = embedding_dir / "vectors.npz"
    manifest_path = embedding_dir / "manifest.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors file not found: {vectors_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest = load_json(manifest_path)

    with np.load(vectors_path) as data:
        text_vectors = data["text_vectors"]
        image_vectors = data["image_vectors"]

    text_records = manifest.get("text_records", [])
    image_records = manifest.get("image_records", [])

    manager = QdrantManager(database_path=args.database)
    try:
        # Create collections
        text_dim = int(text_vectors.shape[1]) if text_vectors.ndim > 1 and text_vectors.shape[0] > 0 else 1024
        image_dim = int(image_vectors.shape[1]) if image_vectors.ndim > 1 and image_vectors.shape[0] > 0 else 768

        manager.create_collection("text_chunks", vector_size=text_dim)
        manager.create_collection("document_images", vector_size=image_dim)

        # Index text chunks
        text_indexer = TextIndexer(manager.client)
        indexed_texts = text_indexer.index(
            collection_name="text_chunks",
            vectors=text_vectors,
            records=text_records,
        )
        print(f"Indexed {indexed_texts} text chunks.")

        # Index images
        image_indexer = ImageIndexer(manager.client)
        indexed_images = image_indexer.index(
            collection_name="document_images",
            vectors=image_vectors,
            records=image_records,
        )
        print(f"Indexed {indexed_images} images.")

        print("\nIndexing completed successfully.")
    finally:
        manager.close()


if __name__ == "__main__":
    main()

