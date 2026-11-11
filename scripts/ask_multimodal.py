"""Multimodal RAG Orchestrator (Phase 8).

Routes user requests between:
1. PDF retrieval (Qdrant + SigLIP 2 / BGE-M3)
2. Question answering (Qwen2.5-VL)
3. AI image generation (Stable Diffusion Turbo)
4. Combined multimodal operations
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv(repo_root / ".env")
load_dotenv()

from src.context.context_builder import ContextBuilder
from src.generation.image_generator import ImageGenerator
from src.generation.image_prompt_builder import ImagePromptBuilder
from src.generation.request_router import RequestRouter
from src.generation.visual_response_builder import VisualResponseBuilder
from src.llm.answer_generator import AnswerGenerator
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline
from src.vectorstore.qdrant_manager import QdrantManager


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute() and not p.exists() and (repo_root / path_str).exists():
        return repo_root / path_str
    return p


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal RAG with retrieval, question answering, and AI image generation."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User question or image generation prompt",
    )
    parser.add_argument(
        "--operation",
        choices=["auto", "qa", "retrieve", "generate", "diagram"],
        default="auto",
        help="Force operation: 'qa', 'retrieve', 'generate', 'diagram', or 'auto' (default)",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Optional document ID filter for retrieval",
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
        help="Maximum PDF images to retrieve",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=3,
        help="Maximum PDF images to attach to Qwen prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Diffusion steps for image generation (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image generation (default: 42)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast evidence-grounded QA synthesis (recommended for CPU execution)",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        default=True,
        help="Use cloud-powered multimodal API (Google Gemini or OpenAI, default: True)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="Cloud provider when running in online mode (default: gemini)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for cloud provider (or set GEMINI_API_KEY / OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--database",
        default="data/processed/qdrant",
        help="Path to local Qdrant database",
    )
    parser.add_argument(
        "--output",
        default="data/generated/responses/latest_visual_response.json",
        help="Path to save final visual response JSON",
    )
    args = parser.parse_args()

    db_path = resolve_path(args.database)
    out_path = Path(args.output)
    if not out_path.is_absolute() and not Path("data").exists() and (repo_root / "data").exists():
        out_path = repo_root / args.output

    print("=" * 65)
    print("MULTIMODAL RAG PIPELINE")
    print("=" * 65)
    print(f"Query: {args.query}\n")

    # Step 1: Request Routing & Operation Selection
    if args.operation == "qa":
        routing = {
            "intent": "answer",
            "retrieve_pdf_images": False,
            "generate_new_image": False,
            "answer_question": True,
        }
    elif args.operation == "retrieve":
        routing = {
            "intent": "retrieve_pdf_images",
            "retrieve_pdf_images": True,
            "generate_new_image": False,
            "answer_question": False,
        }
    elif args.operation == "generate":
        routing = {
            "intent": "generate_image",
            "retrieve_pdf_images": False,
            "generate_new_image": True,
            "generate_diagram": False,
            "answer_question": False,
        }
    elif args.operation == "diagram":
        routing = {
            "intent": "generate_diagram",
            "retrieve_pdf_images": False,
            "generate_new_image": False,
            "generate_diagram": True,
            "answer_question": False,
        }
    else:
        router = RequestRouter()
        routing = router.route(args.query)

    intent = routing["intent"]
    print(f"[Router] Intent resolved: '{intent}' (operation mode: {args.operation})")
    print(
        f"         retrieve_pdf_images={routing.get('retrieve_pdf_images', False)} | "
        f"generate_new_image={routing.get('generate_new_image', False)} | "
        f"generate_diagram={routing.get('generate_diagram', False)} | "
        f"answer_question={routing.get('answer_question', False)}"
    )

    generated_images: List[Dict[str, Any]] = []
    retrieved_images: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    answer: Any = None

    # Step 2: Image Generation (if requested)
    if routing["generate_new_image"]:
        print("\n[Generation] Creating new AI illustration...")
        prompt_builder = ImagePromptBuilder()
        enhanced_prompt = prompt_builder.build(args.query)

        generator = ImageGenerator()
        try:
            gen_meta = generator.generate(
                prompt=enhanced_prompt,
                num_inference_steps=args.steps,
                seed=args.seed,
            )
            generated_images.append(gen_meta)
        finally:
            generator.unload()
            del generator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Step 3: PDF Retrieval (if requested)
    retrieval_results = None
    if routing["retrieve_pdf_images"] or routing.get("answer_question", False) or routing.get("generate_diagram", False):
        print("\n[Retrieval] Searching Qdrant database...")
        manager = QdrantManager(database_path=str(db_path))
        try:
            pipeline = MultimodalRetrievalPipeline(manager.client)
            img_limit = args.image_limit if routing["retrieve_pdf_images"] else 0
            retrieval_results = pipeline.search_multimodal(
                query=args.query,
                text_limit=args.text_limit,
                image_limit=img_limit,
                document_id=args.document_id,
            )
        finally:
            manager.close()

        # Extract retrieved images ONLY when specifically requested
        if routing["retrieve_pdf_images"]:
            raw_images = retrieval_results.get("image_results", [])
            for img in raw_images:
                norm_path = str(img.get("image_path", "")).replace("\\", "/")
                retrieved_images.append({
                    "image_id": img.get("image_id", img.get("id")),
                    "image_path": norm_path,
                    "image_type": "retrieved_from_pdf",
                    "page_number": img.get("page_number"),
                    "score": img.get("score"),
                    "surrounding_text": img.get("surrounding_text"),
                    "document_id": img.get("document_id"),
                })

        # Memory management: unload retrieval pipeline before loading generation models
        del pipeline
        del manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 4: Context Building, QA & Diagram Synthesis
    context = None
    if (routing.get("answer_question", False) or routing.get("generate_diagram", False)) and retrieval_results is not None:
        print("\n[Context] Organizing evidence from PDF...")
        context_builder = ContextBuilder(
            enable_image_descriptions=True,
            max_images_to_describe=0,  # Pass raw images directly to Qwen
        )
        context = context_builder.build(
            query=args.query,
            retrieval_results=retrieval_results,
        )

    # 4a: Question Answering
    if routing.get("answer_question", False) and context is not None:
        print("\n[QA Engine] Generating evidence-grounded answer...")
        answer_gen = AnswerGenerator(
            max_images=args.max_images,
            online_mode=True,
            cloud_provider=args.provider,
            api_key=args.api_key,
        )
        try:
            raw_qa = answer_gen.generate(query=args.query, context=context)
            answer = raw_qa.get("answer")
            sources = raw_qa.get("sources", [])
        finally:
            del answer_gen
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 4b: Diagram Synthesis (Mermaid.js)
    diagram_payload = None
    if routing.get("generate_diagram", False) and context is not None:
        print("\n[Diagram Generator] Synthesizing grounded Mermaid.js diagram...")
        from src.diagram_generator.diagram_service import generate_diagram
        from src.llm.cloud_client import CloudMultimodalClient

        q_lower = args.query.lower()
        if "flowchart" in q_lower:
            diag_type = "flowchart"
        elif "mindmap" in q_lower or "mind map" in q_lower:
            diag_type = "mindmap"
        elif "sequence diagram" in q_lower or "sequence" in q_lower:
            diag_type = "sequenceDiagram"
        elif "class diagram" in q_lower or "class" in q_lower:
            diag_type = "classDiagram"
        elif "er diagram" in q_lower or "erd" in q_lower:
            diag_type = "erDiagram"
        elif "state diagram" in q_lower:
            diag_type = "stateDiagram"
        elif "timeline" in q_lower:
            diag_type = "timeline"
        elif "architecture" in q_lower:
            diag_type = "architecture"
        else:
            diag_type = "auto"

        diag_direction = "LR" if any(w in q_lower for w in ["left to right", "horizontal", " lr"]) else "TD"

        llm = CloudMultimodalClient(provider=args.provider, api_key=args.api_key)
        diag_res = generate_diagram(
            user_query=args.query,
            retrieved_context=context,
            diagram_type=diag_type,
            direction=diag_direction,
            llm=llm,
        )

        if diag_res.get("success"):
            diagram_payload = {
                "diagram_type": diag_res.get("diagram_type", diag_type),
                "mermaid_code": diag_res.get("mermaid_code", ""),
                "repair_attempts": diag_res.get("repair_attempts", 0),
            }
            if not sources:
                sources = diag_res.get("sources", [])
            if not answer:
                answer = f"Here is the technical diagram synthesized from **{args.document_id or 'the document'}**:"
        else:
            err_msg = diag_res.get("error") or "Failed to generate diagram from document context."
            print(f"[Diagram Generator] Notice: {err_msg}")
            if not answer:
                answer = f"⚠️ Notice: {err_msg}"

    elif intent == "retrieve_pdf_images":
        # Construct source list from retrieved images
        for r_img in retrieved_images:
            sources.append({
                "type": "image",
                "id": r_img.get("image_id"),
                "document_id": r_img.get("document_id"),
                "page_number": r_img.get("page_number"),
                "image_path": r_img.get("image_path"),
            })

    # Step 5: Visual Response Formatting
    print("\n[Response Builder] Compiling multimodal response payload...")
    response_builder = VisualResponseBuilder()
    final_response = response_builder.build(
        query=args.query,
        intent=intent,
        answer=answer,
        retrieved_images=retrieved_images,
        generated_images=generated_images,
        diagram=diagram_payload,
        sources=sources,
    )

    save_json(final_response, out_path)
    print(f"Response successfully saved to: {out_path}")

    # Step 6: Console Summary Presentation
    print("\n" + "=" * 65)
    print("MULTIMODAL RESPONSE SUMMARY")
    print("=" * 65)
    print(f"Intent: {intent}")

    if answer:
        print("\n--- Answer ---")
        print(answer)

    if retrieved_images:
        print(f"\n--- Retrieved PDF Images ({len(retrieved_images)}) ---")
        for idx, img in enumerate(retrieved_images, 1):
            print(f"  [{idx}] Page {img.get('page_number')}: {img.get('image_path')} (score: {img.get('score', 0):.4f})")

    if generated_images:
        print(f"\n--- AI Generated Images ({len(generated_images)}) ---")
        for idx, img in enumerate(generated_images, 1):
            print(f"  [{idx}] Path : {img.get('image_path')}")
            print(f"      Model: {img.get('model')} (steps: {img.get('steps')}, seed: {img.get('seed')})")

    print("\n--- Statistics ---")
    for k, v in final_response["statistics"].items():
        print(f"  {k}: {v}")
    print("=" * 65)


if __name__ == "__main__":
    main()

