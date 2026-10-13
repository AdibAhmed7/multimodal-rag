import json
import sys
from pathlib import Path
import numpy as np


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/test_embeddings.py data/processed/embeddings/example"
        )

    output_dir = Path(sys.argv[1])
    vectors_path = output_dir / "vectors.npz"
    manifest_path = output_dir / "manifest.json"

    assert vectors_path.is_file(), f"vectors.npz is missing in {output_dir}"
    assert manifest_path.is_file(), f"manifest.json is missing in {output_dir}"

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    with np.load(vectors_path) as data:
        text_vectors = data["text_vectors"]
        image_vectors = data["image_vectors"]

    assert text_vectors.ndim == 2, f"Expected 2D text vectors, got {text_vectors.ndim}D"
    assert image_vectors.ndim == 2, f"Expected 2D image vectors, got {image_vectors.ndim}D"
    assert text_vectors.shape[0] == manifest["text_vector_count"], (
        f"Text vector count mismatch: {text_vectors.shape[0]} vs {manifest['text_vector_count']}"
    )
    assert image_vectors.shape[0] == manifest["image_vector_count"], (
        f"Image vector count mismatch: {image_vectors.shape[0]} vs {manifest['image_vector_count']}"
    )
    assert len(manifest["text_records"]) == len(text_vectors), (
        "Manifest text records count does not match text vectors count"
    )
    assert len(manifest["image_records"]) == len(image_vectors), (
        "Manifest image records count does not match image vectors count"
    )

    print("\nEmbedding validation passed.")
    print(f"Text vectors:  {text_vectors.shape}")
    print(f"Image vectors: {image_vectors.shape}")
    print(f"Text model:    {manifest.get('text_model')}")
    print(f"Image model:   {manifest.get('image_model')}")


if __name__ == "__main__":
    main()
