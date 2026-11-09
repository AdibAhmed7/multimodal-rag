import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vectorstore.qdrant_manager import QdrantManager
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline
from src.context.context_builder import ContextBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build multimodal context from PDF retrieval."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User question",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Optional document ID to restrict search",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=5,
        help="Maximum number of text chunks to retrieve",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=5,
        help="Maximum number of images to retrieve",
    )
    parser.add_argument(
        "--max-images-to-describe",
        type=int,
        default=3,
        help="Maximum images to describe with Qwen vision model",
    )
    parser.add_argument(
        "--skip-image-descriptions",
        action="store_true",
        help="Skip vision-language model image description",
    )
    parser.add_argument(
        "--database",
        default="data/processed/qdrant",
        help="Path to local Qdrant database",
    )
    parser.add_argument(
        "--output",
        default="data/processed/contexts/latest_context.json",
        help="Path where generated context JSON will be saved",
    )
    args = parser.parse_args()

    manager = QdrantManager(database_path=args.database)
    try:
        pipeline = MultimodalRetrievalPipeline(manager.client)

        print("\nRetrieving relevant information...")
        retrieval_results = pipeline.search_multimodal(
            query=args.query,
            text_limit=args.text_limit,
            image_limit=args.image_limit,
            document_id=args.document_id,
        )

        print("Building structured context...")
        builder = ContextBuilder(
            enable_image_descriptions=not args.skip_image_descriptions,
            max_images_to_describe=args.max_images_to_describe,
        )
        context_package = builder.build(
            query=args.query,
            retrieval_results=retrieval_results,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(context_package, f, indent=2)

        print("\nContext building completed.")
        print(f"Statistics:\n{json.dumps(context_package['statistics'], indent=2)}")
        print(f"\nSaved context to: {output_path.resolve()}")

    finally:
        manager.close()


if __name__ == "__main__":
    main()

