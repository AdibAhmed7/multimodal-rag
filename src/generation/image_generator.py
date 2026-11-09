"""AI Image Generator using Stable Diffusion Turbo with fallback support.

Generates visual diagrams and illustrations based on user prompts and PDF context.
Handles device detection, memory unloading, and synthetic diagram fallback
when running in restricted or CPU-constrained offline environments.
"""

import gc
import logging
import socket
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Wrapper around diffusion models (SD-Turbo) with lifecycle and memory management."""

    def __init__(
        self,
        model_name: str = "stabilityai/sd-turbo",
        output_dir: str = "data/generated/images",
        allow_fallback: bool = True,
    ):
        self.model_name = model_name
        if not Path(output_dir).is_absolute() and not Path("data").exists() and Path("multimodal-rag/data").exists():
            self.output_dir = Path("multimodal-rag") / output_dir
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.allow_fallback = allow_fallback
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.is_fallback = False

    def load_model(self) -> None:
        """Loads the diffusion pipeline into memory or configures the fallback."""
        if self.pipeline is not None or self.is_fallback:
            return

        print(f"Loading Image Generator ({self.model_name}) on {self.device}...")

        # Probe network connectivity quickly to avoid long huggingface timeouts
        has_network = False
        try:
            with socket.create_connection(("huggingface.co", 443), timeout=1.5):
                has_network = True
        except OSError:
            has_network = False

        from diffusers import AutoPipelineForText2Image

        try:
            # 1. Try local cache first
            if self.device == "cuda":
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=True,
                )
                self.pipeline.to("cuda")
            else:
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
                self.pipeline.to("cpu")
            print("Stable Diffusion Turbo loaded successfully from local cache.")
        except Exception as e_local:
            if not has_network:
                if self.allow_fallback:
                    print("Notice: SD-Turbo weights not cached and network is offline.")
                    print("Engaging diagram synthesizer fallback for image generation.")
                    self.is_fallback = True
                    return
                raise RuntimeError(
                    f"SD-Turbo model ({self.model_name}) is not cached locally and network is offline."
                ) from e_local

            # 2. Try online download if network is reachable
            try:
                if self.device == "cuda":
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        variant="fp16",
                    )
                    self.pipeline.to("cuda")
                else:
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                    )
                    self.pipeline.to("cpu")
                print("Stable Diffusion Turbo downloaded and initialized.")
            except Exception as e_online:
                if self.allow_fallback:
                    print(f"Notice: Failed to load diffusion model online ({e_online}). Engaging fallback.")
                    self.is_fallback = True
                    return
                raise RuntimeError(f"Failed to load diffusion pipeline: {e_online}") from e_online

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 2,
        guidance_scale: float = 0.0,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
    ) -> Dict[str, Any]:
        """Generates an image from a prompt and saves it to output_dir.

        Args:
            prompt: Text prompt describing the desired image.
            num_inference_steps: Diffusion denoising steps (default: 2 for SD-Turbo).
            guidance_scale: Classifier-free guidance scale (0.0 for SD-Turbo).
            seed: Random seed for reproducibility.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Dictionary containing image metadata and file path.
        """
        self.load_model()
        image_id = str(uuid.uuid4())
        image_path = self.output_dir / f"{image_id}.png"

        if not self.is_fallback and self.pipeline is not None:
            generator = (
                torch.Generator(device=self.device).manual_seed(seed)
                if seed is not None
                else None
            )
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=width,
                height=height,
            )
            image = result.images[0]
        else:
            # Fallback: Generate an educational diagram canvas using PIL
            image = self._create_diagram_fallback(
                prompt=prompt,
                width=width,
                height=height,
                seed=seed,
            )

        image.save(image_path)
        print(f"Generated image saved to: {image_path}")

        norm_path = str(image_path).replace("\\", "/")
        return {
            "image_id": image_id,
            "image_path": norm_path,
            "image_type": "ai_generated",
            "model": self.model_name,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "seed": seed,
            "is_fallback": self.is_fallback,
        }

    def _create_diagram_fallback(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
    ) -> Image.Image:
        """Creates a stylized educational diagram canvas when offline or on CPU."""
        img = Image.new("RGB", (width, height), color=(15, 23, 42))  # Slate 900
        draw = ImageDraw.Draw(img)

        # Draw subtle grid lines
        grid_color = (30, 41, 59)
        for x in range(0, width, 32):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
        for y in range(0, height, 32):
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)

        # Header bar
        draw.rectangle([(20, 20), (width - 20, 75)], fill=(30, 58, 138), outline=(59, 130, 246), width=2)
        draw.text((35, 30), "AI GENERATED CONCEPT ILLUSTRATION", fill=(255, 255, 255))
        draw.text((35, 52), f"Model: {self.model_name} | Seed: {seed}", fill=(147, 197, 253))

        # Main diagram components (Flow: Input -> Processing -> Output)
        box_color = (30, 41, 59)
        border_color = (56, 189, 248)  # Cyan 400

        # Step 1 box
        draw.rounded_rectangle([(40, 110), (160, 220)], radius=8, fill=box_color, outline=border_color, width=2)
        draw.text((50, 125), "[ Input Layer ]", fill=(255, 255, 255))
        draw.text((50, 150), "- Features", fill=(203, 213, 225))
        draw.text((50, 175), "- Dimensions", fill=(203, 213, 225))

        # Arrow 1 -> 2
        draw.line([(160, 165), (195, 165)], fill=(244, 63, 94), width=3)
        draw.polygon([(195, 160), (205, 165), (195, 170)], fill=(244, 63, 94))

        # Step 2 box
        draw.rounded_rectangle([(205, 95), (345, 235)], radius=8, fill=(30, 41, 59), outline=(168, 85, 247), width=2)
        draw.text((215, 110), "[ Hidden / RAG ]", fill=(255, 255, 255))
        draw.text((215, 140), "* Vector Fusion", fill=(216, 180, 254))
        draw.text((215, 165), "* Multimodal Attn", fill=(216, 180, 254))
        draw.text((215, 190), "* Representation", fill=(216, 180, 254))

        # Arrow 2 -> 3
        draw.line([(345, 165), (380, 165)], fill=(244, 63, 94), width=3)
        draw.polygon([(380, 160), (390, 165), (380, 170)], fill=(244, 63, 94))

        # Step 3 box
        draw.rounded_rectangle([(390, 110), (475, 220)], radius=8, fill=box_color, outline=(34, 197, 94), width=2)
        draw.text((398, 125), "[ Output ]", fill=(255, 255, 255))
        draw.text((398, 150), "> Grounding", fill=(187, 247, 208))
        draw.text((398, 175), "> Predictions", fill=(187, 247, 208))

        # Prompt description box at bottom
        draw.rectangle([(20, 260), (width - 20, height - 25)], fill=(15, 23, 42), outline=(100, 116, 139), width=1)
        draw.text((32, 275), "PROMPT SPECIFICATION:", fill=(250, 204, 21))

        # Truncate / wrap prompt text cleanly
        clean_prompt = prompt.replace("\n", " ").strip()
        lines = []
        words = clean_prompt.split()
        current_line = []
        for word in words:
            if len(" ".join(current_line + [word])) <= 52:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
            if len(lines) >= 6:
                break
        if current_line and len(lines) < 6:
            lines.append(" ".join(current_line))

        y_offset = 305
        for line in lines:
            draw.text((32, y_offset), line, fill=(226, 232, 240))
            y_offset += 24

        return img

    def unload(self) -> None:
        """Releases the diffusion pipeline from memory and clears caches."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.is_fallback = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("ImageGenerator unloaded and GPU/RAM cache cleared.")
