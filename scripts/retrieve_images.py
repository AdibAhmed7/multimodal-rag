"""Dedicated CLI tool for independent PDF Image Retrieval (Operation 2).

Searches the Qdrant document_images collection using SigLIP 2 cross-modal embeddings
and outputs the original PDF figures, page numbers, and file paths without invoking LLMs.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline
from src.vectorstore.qdrant_manager import QdrantManager


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute() and not p.exists() and (repo_root / path_str).exists():
        return repo_root / path_str
    return p


def save_json(data, output_path):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve original images and diagrams from indexed PDFs."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Text query to search for diagrams and figures in the PDF",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Optional document ID to restrict search",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum images to retrieve (default: 4)",
    )
    parser.add_argument(
        "--database",
        default="data/processed/qdrant",
        help="Path to local Qdrant database",
    )
    parser.add_argument(
        "--output",
        default="data/generated/responses/latest_retrieved_images.json",
        help="Path to save retrieval results JSON",
    )
    args = parser.parse_args()

    db_path = resolve_path(args.database)

    print("=" * 65)
    print("PDF IMAGE RETRIEVAL ENGINE")
    print("=" * 65)
    print(f"Query: {args.query}\n")

    manager = QdrantManager(database_path=str(db_path))
    try:
        pipeline = MultimodalRetrievalPipeline(manager.client)
        results = pipeline.search_images(
            query=args.query,
            limit=args.limit,
            document_id=args.document_id,
        )
    finally:
        manager.close()

    formatted_results = []
    print(f"Retrieved {len(results)} images from PDF:\n" + "-" * 65)
    for idx, img in enumerate(results, start=1):
        norm_path = str(img.get("image_path", "")).replace("\\", "/")
        item = {
            "image_id": img.get("id", img.get("image_id")),
            "image_path": norm_path,
            "image_type": "retrieved_from_pdf",
            "page_number": img.get("page_number"),
            "score": float(img.get("score", 0.0)),
            "document_id": img.get("document_id"),
            "surrounding_text": img.get("surrounding_text"),
        }
        formatted_results.append(item)
        print(f"  [{idx}] Page {item['page_number']}: {item['image_path']} (score: {item['score']:.4f})")
        if item.get("surrounding_text"):
            print(f"      Context: {item['surrounding_text'][:90]}...")
    print("-" * 65)

    payload = {
        "query": args.query,
        "operation": "image_retrieval",
        "retrieved_images": formatted_results,
        "count": len(formatted_results),
    }

    out_file = resolve_path(args.output)
    save_json(payload, out_file)
    print(f"Results saved to: {out_file}\n" + "=" * 65)


if __name__ == "__main__":
    main()

