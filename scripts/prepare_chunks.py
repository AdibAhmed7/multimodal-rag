import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.chunking.semantic_chunker import SemanticChunker


def main():
    parser = argparse.ArgumentParser(
        description="Prepare text chunks and image records from Phase 1 output."
    )
    parser.add_argument(
        "metadata_json",
        help="Path to the Phase 1 metadata JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/chunks",
        help="Directory in which to save Phase 2 outputs.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata_json)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata JSON not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    chunker = SemanticChunker()
    result = chunker.process_document(document)

    doc_id = document.get("document_id", metadata_path.stem)
    output_base = Path(args.output_dir)
    if output_base.name == doc_id:
        doc_dir = output_base
    else:
        doc_dir = output_base / doc_id

    doc_dir.mkdir(parents=True, exist_ok=True)

    text_chunks_path = doc_dir / "text_chunks.json"
    image_records_path = doc_dir / "image_records.json"
    document_chunks_path = doc_dir / "document_chunks.json"

    with text_chunks_path.open("w", encoding="utf-8") as f:
        json.dump(result["text_chunks"], f, indent=2)

    with image_records_path.open("w", encoding="utf-8") as f:
        json.dump(result["image_records"], f, indent=2)

    with document_chunks_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Phase 2 preparation complete.")
    print(f"Document: {result.get('filename', metadata_path.name)}")
    print(f"Text chunks: {len(result['text_chunks'])}")
    print(f"Image records: {len(result['image_records'])}")
    print(f"Output directory: {doc_dir.as_posix()}")


if __name__ == "__main__":
    main()
