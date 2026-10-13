import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.embeddings.text_embeddings import TextEmbeddingModel
from src.embeddings.image_embeddings import ImageEmbeddingModel


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text and image embeddings from Phase 2 chunk outputs."
    )
    parser.add_argument(
        "chunk_dir",
        help="Directory containing Phase 2 JSON files (text_chunks.json, image_records.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/embeddings",
        help="Directory in which to save generated vector files.",
    )
    parser.add_argument(
        "--text-model",
        default="BAAI/bge-m3",
        help="Hugging Face model name for text embeddings.",
    )
    parser.add_argument(
        "--image-model",
        default="google/siglip2-base-patch16-224",
        help="Hugging Face model name for image embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    text_path = chunk_dir / "text_chunks.json"
    image_path = chunk_dir / "image_records.json"

    if not text_path.is_file() or not image_path.is_file():
        raise FileNotFoundError(
            f"Required Phase 2 chunk files not found in {chunk_dir}. "
            "Ensure 'text_chunks.json' and 'image_records.json' exist."
        )

    text_records = read_json(text_path)
    image_records = read_json(image_path)

    # Determine document_id
    doc_id = chunk_dir.name
    if text_records and "document_id" in text_records[0]:
        doc_id = text_records[0]["document_id"]

    output_base = Path(args.output_dir)
    doc_output_dir = output_base if output_base.name == doc_id else output_base / doc_id
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate text embeddings (BGE-M3)
    print("\n[1/2] Generating BGE-M3 text embeddings...")
    text_model = TextEmbeddingModel(model_name=args.text_model)
    texts = [record["content"] for record in text_records]
    text_vectors = text_model.encode(texts, batch_size=args.batch_size)
    print(f"Text vector shape: {text_vectors.shape}")

    # 2. Generate image embeddings (SigLIP 2)
    print("\n[2/2] Generating SigLIP 2 image embeddings...")
    image_model = ImageEmbeddingModel(model_name=args.image_model)
    image_paths = []
    for record in image_records:
        img_path = Path(record["image_path"])
        if not img_path.is_absolute() and not img_path.exists():
            candidate = repo_root / img_path
            if candidate.exists():
                img_path = candidate
        image_paths.append(img_path)

    image_vectors = image_model.encode_images(image_paths, batch_size=args.batch_size)
    print(f"Image vector shape: {image_vectors.shape}")

    # Save vectors to .npz
    vectors_file = doc_output_dir / "vectors.npz"
    np.savez_compressed(
        vectors_file,
        text_vectors=text_vectors,
        image_vectors=image_vectors,
    )

    # Build and save manifest
    manifest = {
        "document_id": doc_id,
        "text_model": args.text_model,
        "text_vector_count": int(text_vectors.shape[0]),
        "text_dimension": int(text_vectors.shape[1]) if text_vectors.ndim > 1 and text_vectors.shape[0] > 0 else 0,
        "image_model": args.image_model,
        "image_vector_count": int(image_vectors.shape[0]),
        "image_dimension": int(image_vectors.shape[1]) if image_vectors.ndim > 1 and image_vectors.shape[0] > 0 else 0,
        "text_records": text_records,
        "image_records": image_records,
    }

    manifest_file = doc_output_dir / "manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nEmbedding generation complete.")
    print(f"Vectors:  {vectors_file.as_posix()}")
    print(f"Manifest: {manifest_file.as_posix()}")


if __name__ == "__main__":
    main()
