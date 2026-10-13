import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/test_chunking.py "
            "data/processed/chunks/example"
        )

    output_dir = Path(sys.argv[1])
    text_path = output_dir / "text_chunks.json"
    image_path = output_dir / "image_records.json"

    assert text_path.is_file(), "Text chunks file is missing."
    assert image_path.is_file(), "Image records file is missing."

    text_chunks = json.loads(
        text_path.read_text(encoding="utf-8")
    )
    image_records = json.loads(
        image_path.read_text(encoding="utf-8")
    )

    assert text_chunks, "No text chunks were produced."

    for chunk in text_chunks:
        assert chunk["chunk_id"]
        assert chunk["document_id"]
        assert chunk["page_number"] >= 1
        assert chunk["content"].strip()

    for image in image_records:
        assert image["image_id"]
        assert image["page_number"] >= 1
        assert image["document_id"]
        image_file = Path(image["image_path"])
        assert image_file.is_file(), (
            f"Image file missing: {image_file}"
        )
        for chunk_id in image["related_text_chunk_ids"]:
            assert any(
                chunk["chunk_id"] == chunk_id for chunk in text_chunks
            ), f"Unknown related text chunk: {chunk_id}"

    print("Phase 2 validation passed.")
    print(f"Validated {len(text_chunks)} text chunks.")
    print(f"Validated {len(image_records)} image records.")


if __name__ == "__main__":
    main()
