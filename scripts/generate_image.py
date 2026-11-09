import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.generation.image_generator import ImageGenerator
from src.generation.image_prompt_builder import ImagePromptBuilder


def save_json(data, output_path):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone AI image generator using Stable Diffusion Turbo."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt describing the desired image",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of diffusion denoising steps (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="data/generated/responses/latest_generated_image.json",
        help="Path to save generation metadata JSON",
    )
    args = parser.parse_args()

    print(f"User Prompt: {args.prompt}")
    builder = ImagePromptBuilder()
    enhanced_prompt = builder.build(args.prompt)
    print("\nEnhanced Prompt:")
    print("-" * 50)
    print(enhanced_prompt)
    print("-" * 50)

    generator = ImageGenerator()
    try:
        metadata = generator.generate(
            prompt=enhanced_prompt,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
    finally:
        generator.unload()

    save_json(metadata, args.output)

    print("\nImage Generation Complete!")
    print(f"Generated Image : {metadata['image_path']}")
    print(f"Metadata Saved  : {args.output}")


if __name__ == "__main__":
    main()
