import argparse
import gc
import json
import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vectorstore.qdrant_manager import QdrantManager
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline
from src.context.context_builder import ContextBuilder
from src.llm.answer_generator import AnswerGenerator
from src.llm.response_formatter import ResponseFormatter


def save_json(data, output_path):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about indexed PDFs using multimodal RAG."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question to ask about the PDF document(s)",
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
        help="Maximum text chunks to retrieve",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=4,
        help="Maximum document images to retrieve",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=3,
        help="Maximum images to attach to the answering prompt",
    )
    parser.add_argument(
        "--database",
        default="data/processed/qdrant",
        help="Path to local Qdrant database",
    )
    parser.add_argument(
        "--output",
        default="data/generated/responses/latest_response.json",
        help="Path to save response JSON",
    )
    args = parser.parse_args()

    # Step 1: Multimodal Retrieval
    print("[1/4] Retrieving PDF evidence...")
    manager = QdrantManager(database_path=args.database)
    try:
        pipeline = MultimodalRetrievalPipeline(manager.client)
        retrieval_results = pipeline.search_multimodal(
            query=args.query,
            text_limit=args.text_limit,
            image_limit=args.image_limit,
            document_id=args.document_id,
        )
    finally:
        manager.close()

    # Explicit memory release before loading LLM
    del pipeline
    del manager
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Context Building
    print("\n[2/4] Building context...")
    # Original images are passed to Qwen directly, so on-the-fly captioning can be skipped
    context_builder = ContextBuilder(
        enable_image_descriptions=True,
        max_images_to_describe=0,
    )
    context = context_builder.build(
        query=args.query,
        retrieval_results=retrieval_results,
    )

    # Step 3: Answer Generation
    print("\n[3/4] Generating answer with Qwen...")
    generator = AnswerGenerator(max_images=args.max_images)
    raw_result = generator.generate(query=args.query, context=context)

    # Step 4: Formatting and Output
    print("\n[4/4] Formatting response...")
    formatted_response = ResponseFormatter.format(raw_result)
    save_json(formatted_response, args.output)

    # Display Answer and Citations
    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(formatted_response["answer"])

    print("\n" + "=" * 60)
    print("SOURCE PAGES")
    print("=" * 60)
    pages = formatted_response["sources"]["pages"]
    if pages:
        for p in pages:
            print(f"Document: {p['document_id']} | Page: {p['page_number']}")
    else:
        print("No specific source pages cited.")

    print(f"\nSaved response:\n{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

