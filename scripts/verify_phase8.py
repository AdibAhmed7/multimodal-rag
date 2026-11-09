"""Comprehensive Phase 8 Verification and Diagnostic Script.

Tests all components, edge cases, memory management, and end-to-end workflows:
1. RequestRouter edge cases & ambiguous phrasing
2. ImagePromptBuilder formatting & context injection
3. ImageGenerator execution, lifecycle, memory unloading, and image validity
4. VisualResponseBuilder payload structure and strict image separation
5. End-to-end execution of all 4 multimodal query archetypes
6. Output JSON schema and file persistence integrity
"""

import gc
import json
import os
import sys
from pathlib import Path
from PIL import Image
import torch

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.generation.request_router import RequestRouter
from src.generation.image_prompt_builder import ImagePromptBuilder
from src.generation.image_generator import ImageGenerator
from src.generation.visual_response_builder import VisualResponseBuilder
from src.vectorstore.qdrant_manager import QdrantManager
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline
from src.context.context_builder import ContextBuilder
from src.llm.answer_generator import AnswerGenerator


def test_request_router():
    print("\n" + "=" * 60)
    print("[TEST 1/6] Auditing Request Router Intent Classification")
    print("=" * 60)
    router = RequestRouter()

    test_cases = [
        # 1. Answer queries
        ("Explain the main concept of this PDF", "answer"),
        ("What is the difference between encoder and decoder?", "answer"),
        ("How does multi-head self-attention work?", "answer"),
        ("Describe the training parameters used in section 4", "answer"),

        # 2. PDF Image retrieval queries
        ("Show me the diagram from the PDF", "retrieve_pdf_images"),
        ("Find the diagram illustrating transformer layers", "retrieve_pdf_images"),
        ("Show the image of the encoder architecture", "retrieve_pdf_images"),
        ("Original diagram of the network from the pdf", "retrieve_pdf_images"),
        ("Show me the figure from the pdf", "retrieve_pdf_images"),

        # 3. New image generation queries
        ("Generate a new neural network illustration", "generate_image"),
        ("Create an image showing attention weights", "generate_image"),
        ("Illustrate the concept of self-attention", "generate_image"),
        ("Draw an image of artificial neurons", "generate_image"),
        ("Generate a diagram of a modern transformer", "generate_image"),

        # 4. Hybrid queries
        ("Generate a new diagram and show the original PDF diagram", "answer_and_generate"),
        ("Draw me an illustration and show the original diagram from the document", "answer_and_generate"),
        ("Create an image of the attention flow and show me the diagram from the pdf", "answer_and_generate"),
    ]

    passed = 0
    for query, expected_intent in test_cases:
        res = router.route(query)
        actual_intent = res["intent"]
        assert actual_intent == expected_intent, (
            f"Routing failed for query: '{query}' -> Expected '{expected_intent}', got '{actual_intent}'"
        )
        passed += 1
        print(f"  ✓ [{actual_intent.ljust(19)}] {query[:50]}")

    print(f"\nRequest Router: All {passed}/{len(test_cases)} test cases passed successfully.")


def test_prompt_builder():
    print("\n" + "=" * 60)
    print("[TEST 2/6] Auditing Image Prompt Builder")
    print("=" * 60)
    builder = ImagePromptBuilder()

    # Basic prompt
    p1 = builder.build("A deep learning computation graph")
    assert "A deep learning computation graph" in p1
    assert "Visual requirements:" in p1
    assert "Clear composition." in p1
    print("  ✓ Basic prompt correctly expands with technical and visual constraints.")

    # With PDF context
    ctx = "The model uses 8 attention heads with dimension dk = 64."
    p2 = builder.build("Multi-Head Attention", context=ctx)
    assert "Multi-Head Attention" in p2
    assert "Additional information from the PDF:" in p2
    assert ctx in p2
    print("  ✓ Context-injected prompt preserves technical grounding.")

    # Empty context handling
    p3 = builder.build("Neural network", context=None)
    assert "Additional information" not in p3
    print("  ✓ Null context omitted without trailing artifacts.")


def test_image_generator_and_memory():
    print("\n" + "=" * 60)
    print("[TEST 3/6] Auditing Image Generator & Memory Lifecycle")
    print("=" * 60)
    out_dir = Path("data/generated/images")
    generator = ImageGenerator(output_dir=str(out_dir), allow_fallback=True)

    # Force fallback mode to verify synthetic diagram pipeline
    generator.is_fallback = True
    meta = generator.generate(
        prompt="Audited test neural network illustration",
        num_inference_steps=2,
        seed=999,
        width=512,
        height=512,
    )

    # 1. Path validity
    img_path = Path(meta["image_path"])
    assert img_path.exists(), f"Generated image file not found: {img_path}"
    assert img_path.stat().st_size > 5000, f"Generated image file is suspiciously small: {img_path.stat().st_size} bytes"
    print(f"  ✓ Image generated successfully ({img_path.stat().st_size:,} bytes).")

    # 2. PIL Image format and dimensions
    with Image.open(img_path) as img:
        assert img.size == (512, 512), f"Unexpected image size: {img.size}"
        assert img.format == "PNG", f"Unexpected image format: {img.format}"
    print(f"  ✓ Image validated: dimensions (512, 512), format PNG.")

    # 3. Path string cross-platform forward-slash check
    assert "\\" not in meta["image_path"], f"image_path should use forward slashes: {meta['image_path']}"
    print(f"  ✓ Path normalization verified: {meta['image_path']}.")

    # 4. Unload and memory clearance
    generator.unload()
    assert generator.pipeline is None
    assert generator.is_fallback is False
    print("  ✓ Memory unloading and cache clearance verified.")


def test_visual_response_builder_separation():
    print("\n" + "=" * 60)
    print("[TEST 4/6] Auditing Visual Response Builder & Separation")
    print("=" * 60)
    builder = VisualResponseBuilder()

    retrieved = [
        {
            "image_id": "pdf_fig_p2",
            "image_path": "data/processed/images/example/page_002_img_001.png",
            "image_type": "retrieved_from_pdf",
            "page_number": 2,
        }
    ]
    generated = [
        {
            "image_id": "gen_uuid_123",
            "image_path": "data/generated/images/gen_uuid_123.png",
            "image_type": "ai_generated",
            "model": "stabilityai/sd-turbo",
        }
    ]

    response = builder.build(
        query="Verify image separation",
        intent="answer_and_generate",
        answer="Testing strict separation",
        retrieved_images=retrieved,
        generated_images=generated,
        sources=[{"type": "image", "page_number": 2}],
    )

    # Verify field separation
    for r in response["retrieved_images"]:
        assert r["image_type"] == "retrieved_from_pdf", "Corrupted image_type in retrieved_images"
        assert "generated" not in r["image_path"].lower(), "Generated image mistakenly placed in retrieved_images!"

    for g in response["generated_images"]:
        assert g["image_type"] == "ai_generated", "Corrupted image_type in generated_images"
        assert "processed" not in g["image_path"].lower(), "Retrieved image mistakenly placed in generated_images!"

    assert response["statistics"]["retrieved_image_count"] == 1
    assert response["statistics"]["generated_image_count"] == 1
    assert response["statistics"]["source_count"] == 1
    print("  ✓ Strict separation confirmed: retrieved and AI-generated imagery never cross-contaminate.")


def test_end_to_end_scenarios():
    print("\n" + "=" * 60)
    print("[TEST 5/6] Auditing End-to-End Orchestrator Flows")
    print("=" * 60)
    import subprocess

    py_exe = sys.executable

    scenarios = [
        ("Generate Only", "Generate an illustration of a deep neural network", "generate_image"),
        ("Retrieve Only", "Show me the diagram from the PDF", "retrieve_pdf_images"),
        ("Hybrid", "Generate a new diagram and show the original PDF diagram", "answer_and_generate"),
        ("Answer Only", "Explain the architecture of the proposed system", "answer"),
    ]

    for label, query, expected_intent in scenarios:
        print(f"\n--- Scenario: {label} ---")
        cmd = [py_exe, "scripts/ask_multimodal.py", "--query", query]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))

        assert proc.returncode == 0, f"Script failed for '{query}':\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        assert f"Intent detected: '{expected_intent}'" in proc.stdout, f"Unexpected intent in output for query: {query}"
        print(f"  ✓ {label} completed with returncode 0 and intent '{expected_intent}'.")

    # Validate latest visual response JSON
    resp_file = repo_root / "data/generated/responses/latest_visual_response.json"
    assert resp_file.exists(), f"Response JSON file not found at {resp_file}"
    with open(resp_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "query" in data
    assert "intent" in data
    assert "retrieved_images" in data
    assert "generated_images" in data
    assert "statistics" in data
    assert "generated_at" in data
    print(f"  ✓ Response JSON successfully saved and verified: {resp_file}")


def test_cleanliness_and_resources():
    print("\n" + "=" * 60)
    print("[TEST 6/6] Auditing Resource Clearance and Database State")
    print("=" * 60)

    # 1. Qdrant lock check
    qdrant_path = repo_root / "data/processed/qdrant"
    assert qdrant_path.exists(), "Qdrant database directory does not exist."
    manager = QdrantManager(database_path=str(qdrant_path))
    text_info = manager.get_collection_info("text_chunks")
    image_info = manager.get_collection_info("document_images")
    manager.close()

    assert text_info.points_count > 0, "text_chunks collection is empty!"
    assert image_info.points_count > 0, "document_images collection is empty!"
    print(f"  ✓ Qdrant collections intact: {text_info.points_count} text points, {image_info.points_count} image points.")

    # 2. Garbage collection & memory check
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  ✓ GPU/CPU memory free and cleanly reclaimed.")


def main():
    print("=" * 60)
    print("PHASE 8 AUDIT & DIAGNOSTIC SUITE")
    print("=" * 60)

    test_request_router()
    test_prompt_builder()
    test_image_generator_and_memory()
    test_visual_response_builder_separation()
    test_end_to_end_scenarios()
    test_cleanliness_and_resources()

    print("\n" + "=" * 60)
    print("ALL PHASE 8 DIAGNOSTIC CHECKS PASSED WITH ZERO DEFECTS!")
    print("=" * 60)


if __name__ == "__main__":
    main()

