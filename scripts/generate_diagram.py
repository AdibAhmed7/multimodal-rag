"""Command line interface for generating grounded Mermaid.js diagrams from indexed PDFs."""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv(repo_root / ".env")
load_dotenv()

from src.context.context_builder import ContextBuilder
from src.diagram_generator.diagram_service import generate_diagram
from src.diagram_generator.diagram_utils import build_mermaid_html
from src.llm.cloud_client import CloudMultimodalClient
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline
from src.vectorstore.qdrant_manager import QdrantManager


def main():
    parser = argparse.ArgumentParser(description="Generate grounded Mermaid.js diagrams from PDF documents.")
    parser.add_argument("--query", required=True, help="Instruction or request describing the desired diagram.")
    parser.add_argument("--document-id", default=None, help="Target document ID indexed in Qdrant.")
    parser.add_argument(
        "--type",
        default="auto",
        choices=[
            "auto",
            "flowchart",
            "mindmap",
            "sequenceDiagram",
            "classDiagram",
            "erDiagram",
            "stateDiagram",
            "timeline",
            "architecture",
        ],
        help="Target diagram type (default: auto).",
    )
    parser.add_argument("--direction", default="TD", choices=["TD", "LR"], help="Flow direction (TD or LR).")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"], help="Cloud LLM provider.")
    parser.add_argument("--text-limit", type=int, default=6, help="Number of text chunks to retrieve.")
    parser.add_argument("--database", default="data/processed/qdrant", help="Path to Qdrant database.")
    parser.add_argument("--output", default=None, help="Optional output path for .mmd or .html file.")
    args = parser.parse_args()

    print("=" * 65)
    print("AI-POWERED DIAGRAM GENERATOR (MERMAID.JS)")
    print("=" * 65)
    print(f"Query:        {args.query}")
    print(f"Document ID:  {args.document_id or 'All Documents'}")
    print(f"Diagram Type: {args.type} (Direction: {args.direction})")
    print(f"LLM Provider: {args.provider.upper()}\n")

    # Step 1: Retrieval
    print("[1/3] Retrieving relevant document context...")
    manager = QdrantManager(database_path=args.database)
    try:
        pipeline = MultimodalRetrievalPipeline(manager.client)
        retrieval_results = pipeline.search_multimodal(
            query=args.query,
            text_limit=args.text_limit,
            image_limit=0,
            document_id=args.document_id,
        )
    finally:
        manager.close()

    text_chunks = retrieval_results.get("text_results", [])
    print(f"      Retrieved {len(text_chunks)} text chunk(s) from document.")

    # Step 2: Context Organization
    context_builder = ContextBuilder(enable_image_descriptions=False, max_images_to_describe=0)
    context = context_builder.build(query=args.query, retrieval_results=retrieval_results)

    # Step 3: Diagram Generation
    print("\n[2/3] Generating grounded Mermaid syntax via LLM...")
    llm = CloudMultimodalClient(provider=args.provider)
    result = generate_diagram(
        user_query=args.query,
        retrieved_context=context,
        diagram_type=args.type,
        direction=args.direction,
        llm=llm,
    )

    if not result["success"]:
        print(f"\n[Error] Diagram generation failed:\n{result.get('error')}")
        sys.exit(1)

    mermaid_code = result["mermaid_code"]
    print("\n[3/3] Diagram successfully generated and validated!")
    print(f"Resolved Type: {result.get('diagram_type')}\n")
    print("-" * 50)
    print(mermaid_code)
    print("-" * 50)

    # Step 4: Optional Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            import json
            payload = {
                "success": result["success"],
                "diagram_type": result.get("diagram_type"),
                "mermaid_code": mermaid_code,
                "error": result.get("error"),
                "citations": [
                    {
                        "document_id": p.get("document_id"),
                        "page_number": p.get("page_number"),
                        "chunk_count": len(p.get("text_chunks", [])),
                        "preview": (p.get("text_chunks", [""])[0][:180] + "...") if p.get("text_chunks") else ""
                    }
                    for p in context.get("pages", [])
                ],
                "context_text_count": len(text_chunks)
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif out_path.suffix.lower() == ".html":
            html = build_mermaid_html(mermaid_code)
            out_path.write_text(html, encoding="utf-8")
        else:
            out_path.write_text(mermaid_code, encoding="utf-8")
        print(f"\nDiagram saved to: {out_path}")


if __name__ == "__main__":
    main()
