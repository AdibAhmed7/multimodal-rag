"""Verification script for the three independent core operations:
1. PDF Question Answering
2. Original Image Retrieval
3. New Image Generation

Tests both dedicated CLI entry points and orchestrator operation modes.
"""

import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

PYTHON_EXE = sys.executable


def run_command(cmd_args):
    print(f"Running: {' '.join(cmd_args)}")
    proc = subprocess.run(
        cmd_args,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert proc.returncode == 0, f"Command failed with code {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc.stdout


def test_operation_1_pdf_qa():
    print("\n" + "=" * 65)
    print("TESTING OPERATION 1: PDF QUESTION ANSWERING (STANDALONE & ISOLATED)")
    print("=" * 65)

    # 1. Test via dedicated ask_pdf.py
    out_file = repo_root / "data/generated/responses/test_qa_response.json"
    cmd = [
        PYTHON_EXE,
        "scripts/ask_pdf.py",
        "--query",
        "Explain the architecture of the proposed system",
        "--output",
        str(out_file),
    ]
    stdout = run_command(cmd)
    assert "[1/4] Retrieving PDF evidence..." in stdout
    assert "[3/4] Generating answer with Qwen..." in stdout
    assert out_file.exists(), f"Output file missing: {out_file}"

    with open(out_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    assert qa_data.get("answer"), "Answer must not be empty"
    assert "Page" in qa_data["answer"], "Answer should include page citations"
    print("  ✓ Standalone ask_pdf.py successfully answered with citations.")

    # 2. Test via orchestrator with --operation qa
    out_multi = repo_root / "data/generated/responses/test_multi_qa.json"
    cmd_multi = [
        PYTHON_EXE,
        "scripts/ask_multimodal.py",
        "--operation",
        "qa",
        "--query",
        "Explain the transformer architecture",
        "--output",
        str(out_multi),
    ]
    stdout_multi = run_command(cmd_multi)
    assert "Intent resolved: 'answer' (operation mode: qa)" in stdout_multi
    assert "generate_new_image=False" in stdout_multi

    with open(out_multi, "r", encoding="utf-8") as f:
        multi_data = json.load(f)

    assert multi_data["intent"] == "answer"
    assert multi_data["answer"] is not None
    assert multi_data["statistics"]["generated_image_count"] == 0
    print("  ✓ Orchestrator --operation qa strictly executes QA without new image generation.")


def test_operation_2_image_retrieval():
    print("\n" + "=" * 65)
    print("TESTING OPERATION 2: ORIGINAL IMAGE RETRIEVAL (STANDALONE & ISOLATED)")
    print("=" * 65)

    # 1. Test via dedicated retrieve_images.py
    out_file = repo_root / "data/generated/responses/test_retrieved_images.json"
    cmd = [
        PYTHON_EXE,
        "scripts/retrieve_images.py",
        "--query",
        "Find the architecture diagram in the PDF",
        "--output",
        str(out_file),
    ]
    stdout = run_command(cmd)
    assert "PDF IMAGE RETRIEVAL ENGINE" in stdout
    assert out_file.exists(), f"Output file missing: {out_file}"

    with open(out_file, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)

    assert retrieval_data["operation"] == "image_retrieval"
    assert len(retrieval_data["retrieved_images"]) > 0
    for img in retrieval_data["retrieved_images"]:
        assert img["image_type"] == "retrieved_from_pdf"
        assert Path(img["image_path"]).exists(), f"Retrieved image file missing: {img['image_path']}"
    print(f"  ✓ Standalone retrieve_images.py retrieved {len(retrieval_data['retrieved_images'])} original images.")

    # 2. Test via orchestrator with --operation retrieve
    out_multi = repo_root / "data/generated/responses/test_multi_retrieve.json"
    cmd_multi = [
        PYTHON_EXE,
        "scripts/ask_multimodal.py",
        "--operation",
        "retrieve",
        "--query",
        "Show me the diagram from the PDF",
        "--output",
        str(out_multi),
    ]
    stdout_multi = run_command(cmd_multi)
    assert "Intent resolved: 'retrieve_pdf_images' (operation mode: retrieve)" in stdout_multi
    assert "generate_new_image=False" in stdout_multi
    assert "answer_question=False" in stdout_multi

    with open(out_multi, "r", encoding="utf-8") as f:
        multi_data = json.load(f)

    assert multi_data["intent"] == "retrieve_pdf_images"
    assert multi_data["answer"] is None, "QA answer must be None in pure image retrieval mode"
    assert multi_data["statistics"]["generated_image_count"] == 0
    assert multi_data["statistics"]["retrieved_image_count"] > 0
    print("  ✓ Orchestrator --operation retrieve strictly returns PDF images without QA or generation.")


def test_operation_3_image_generation():
    print("\n" + "=" * 65)
    print("TESTING OPERATION 3: NEW IMAGE GENERATION (STANDALONE & ISOLATED)")
    print("=" * 65)

    # 1. Test via dedicated generate_image.py
    out_file = repo_root / "data/generated/responses/test_generated_image.json"
    cmd = [
        PYTHON_EXE,
        "scripts/generate_image.py",
        "--prompt",
        "A neural network flowchart with feedforward layers",
        "--output",
        str(out_file),
    ]
    stdout = run_command(cmd)
    assert "Image Generation Complete!" in stdout
    assert out_file.exists(), f"Output file missing: {out_file}"

    with open(out_file, "r", encoding="utf-8") as f:
        gen_data = json.load(f)

    assert gen_data["image_type"] == "ai_generated"
    assert Path(gen_data["image_path"]).exists()
    print("  ✓ Standalone generate_image.py synthesized new image without retrieval or QA.")

    # 2. Test via orchestrator with --operation generate
    out_multi = repo_root / "data/generated/responses/test_multi_generate.json"
    cmd_multi = [
        PYTHON_EXE,
        "scripts/ask_multimodal.py",
        "--operation",
        "generate",
        "--query",
        "Generate a modern AI architecture diagram",
        "--output",
        str(out_multi),
    ]
    stdout_multi = run_command(cmd_multi)
    assert "Intent resolved: 'generate_image' (operation mode: generate)" in stdout_multi
    assert "retrieve_pdf_images=False" in stdout_multi
    assert "answer_question=False" in stdout_multi

    with open(out_multi, "r", encoding="utf-8") as f:
        multi_data = json.load(f)

    assert multi_data["intent"] == "generate_image"
    assert multi_data["answer"] is None, "QA answer must be None in pure image generation mode"
    assert multi_data["statistics"]["retrieved_image_count"] == 0
    assert multi_data["statistics"]["generated_image_count"] == 1
    print("  ✓ Orchestrator --operation generate strictly synthesizes new images without PDF retrieval.")


def main():
    print("=" * 65)
    print("CORE INDEPENDENT OPERATIONS VALIDATION SUITE")
    print("=" * 65)

    test_operation_1_pdf_qa()
    test_operation_2_image_retrieval()
    test_operation_3_image_generation()

    print("\n" + "=" * 65)
    print("ALL THREE CORE OPERATIONS WORK COMPLETELY INDEPENDENTLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()

