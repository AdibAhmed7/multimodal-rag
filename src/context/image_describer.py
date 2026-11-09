import json
import logging
from pathlib import Path
from typing import Optional, Union
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


class ImageDescriber:
    """Uses Qwen2.5-VL vision-language model to describe figures and diagrams.

    Loads model weights lazily and maintains a persistent JSON cache of descriptions.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        cache_path: str = "data/processed/image_descriptions/cache.json",
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def _load_cache(self) -> dict:
        """Load cached image descriptions from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read image description cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Persist image descriptions to disk."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write image description cache: {e}")

    def _lazy_load_model(self) -> None:
        """Instantiate Qwen2.5-VL model and processor on first use."""
        if self.model is not None:
            return

        print(f"Loading Qwen vision-language model ({self.model_name})...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
                local_files_only=True,
            )
        except Exception:
            # Fall back to downloading if not cached locally
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
            )

        if not torch.cuda.is_available() and hasattr(self.model, "to"):
            self.model.to(self.device)

        self.model.eval()
        print("Qwen model loaded.")

    def describe_image(
        self,
        image_path: Union[str, Path],
        surrounding_text: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate or retrieve a cached textual description for a PDF figure."""
        path_obj = Path(image_path)
        cache_key = path_obj.name

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        if str(image_path) in self.cache:
            return self.cache[str(image_path)]

        if not path_obj.exists():
            return f"[Image file not found: {image_path}]"

        try:
            self._lazy_load_model()

            if not prompt:
                prompt = "Describe this diagram or figure from the document clearly and concisely in 2-3 sentences."
            if surrounding_text and surrounding_text.strip():
                prompt += f"\nContext from the document:\n{surrounding_text.strip()}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(path_obj.resolve())},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            self.cache[cache_key] = output_text
            self._save_cache()
            return output_text

        except Exception as e:
            logger.error(f"Failed to generate description for {image_path}: {e}")
            fallback = f"Visual figure from document page (description generation failed: {e})"
            return fallback

