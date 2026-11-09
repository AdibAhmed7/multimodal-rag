import logging
import re
from typing import Any, Dict, List
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


class QwenModel:
    """Loads and interfaces with Qwen2.5-VL vision-language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        allow_fallback: bool = True,
    ):
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.is_fallback = False

        print(f"Loading Qwen on {self.device}...")
        self._load_model()

    def _load_model(self) -> None:
        """Attempt loading Qwen model weights, with graceful fallback on offline/memory limits."""
        import socket

        has_network = False
        try:
            with socket.create_connection(("huggingface.co", 443), timeout=1.5):
                has_network = True
        except OSError:
            has_network = False

        try:
            # 1. Check local cache first
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device.type == "cuda" else None,
                local_files_only=True,
            )
        except Exception as e_local:
            if not has_network:
                if self.allow_fallback:
                    print("Notice: Qwen weights not in local cache and environment is offline.")
                    print("Engaging evidence-grounded multimodal synthesizer fallback.")
                    self.is_fallback = True
                    return
                raise RuntimeError(
                    f"Qwen model ({self.model_name}) is not cached locally and network is offline."
                ) from e_local

            try:
                # 2. Try online download if network is available
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if self.device.type == "cuda" else None,
                )
            except Exception as e_remote:
                if self.allow_fallback:
                    print(f"Notice: Failed to load Qwen ({e_remote}).")
                    print("Engaging evidence-grounded multimodal synthesizer fallback.")
                    self.is_fallback = True
                    return
                raise RuntimeError(
                    f"Failed to load Qwen model ({self.model_name}): {e_remote}"
                ) from e_remote

        if self.device.type == "cpu" and hasattr(self.model, "to"):
            self.model.to(self.device)

        self.model.eval()
        print("Qwen model loaded successfully.")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 700,
        do_sample: bool = False,
    ) -> str:
        """Generate response from multimodal messages."""
        if self.is_fallback:
            return self._fallback_generate(messages)

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
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return response

    def _fallback_generate(self, messages: List[Dict[str, Any]]) -> str:
        """Heuristic evidence-grounded synthesizer when 3B weights cannot be downloaded/loaded in current environment."""
        user_text = ""
        attached_images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        user_text += item.get("text", "") + "\n"
                    elif item.get("type") == "image":
                        attached_images.append(item.get("image"))
            elif isinstance(content, str):
                user_text += content + "\n"

        # Extract page sections from prompt evidence
        page_pattern = re.compile(r"--- Document:\s*(\S+)\s*\|\s*Page:\s*(\d+)\s*---(.*?)(?=--- Document:|\Z)", re.DOTALL)
        matches = page_pattern.findall(user_text)

        evidence_paragraphs = []
        pages_cited = set()

        for doc_id, page_num, section_text in matches:
            pages_cited.add(page_num)
            # Find chunks
            chunks = re.findall(r"\[Chunk\s+[^\]]+\]:\s*([^\n]+)", section_text)
            for c in chunks:
                if c.strip():
                    evidence_paragraphs.append(f"{c.strip()} [Page {page_num}]")

            # Find image descriptors
            img_descs = re.findall(r"\[Image\s+([^\]]+)\](?:\s+Description:\s*([^\n]+)|\s+Context:\s*([^\n]+))?", section_text)
            for img_id, desc, ctx in img_descs:
                detail = desc or ctx or "visual diagram"
                evidence_paragraphs.append(
                    f"The document includes figure ({img_id}) illustrating {detail.strip()} [Page {page_num}]."
                )

        if not evidence_paragraphs:
            return "The provided document does not contain enough information to answer this question reliably."

        answer_body = "\n\n".join(evidence_paragraphs[:4])
        return answer_body
