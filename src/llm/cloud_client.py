"""Online Cloud Multimodal LLM Client (Phase 10+).

Provides zero-footprint, cloud-accelerated multimodal question answering
via Google Gemini (Gemini 2.5 / 1.5 Flash / Pro) and OpenAI (GPT-4o / GPT-4o-mini).
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from dotenv import load_dotenv

# Automatically load environment variables from current directory or repo root
load_dotenv()
repo_root = Path(__file__).resolve().parent.parent.parent
if (repo_root / ".env").exists():
    load_dotenv(repo_root / ".env")


class CloudMultimodalClient:
    """Dispatches multimodal reasoning requests to Groq, Google Gemini, or OpenAI Cloud APIs."""

    def __init__(
        self,
        provider: str = "groq",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider.lower().strip()
        self.api_key = (
            api_key
            or os.environ.get(f"{self.provider.upper()}_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

        if self.provider == "groq":
            self.model_name = model_name or "openai/gpt-oss-120b"
        elif self.provider == "gemini":
            self.model_name = model_name or "gemini-2.5-flash"
        elif self.provider in ("openai", "gpt"):
            self.provider = "openai"
            self.model_name = model_name or "gpt-4o-mini"
        else:
            raise ValueError(f"Unsupported cloud provider '{provider}'. Choose 'groq', 'gemini', or 'openai'.")

    @classmethod
    def is_configured(cls, provider: str = "groq") -> bool:
        """Check if an API key is present in environment for the specified provider."""
        p = provider.lower()
        if p == "groq":
            return bool(os.environ.get("GROQ_API_KEY"))
        if p == "gemini":
            return bool(os.environ.get("GEMINI_API_KEY"))
        if p in ("openai", "gpt"):
            return bool(os.environ.get("OPENAI_API_KEY"))
        return False

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Optional[List[Union[str, Path]]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
        """Execute multimodal inference on the chosen cloud platform."""
        if not self.api_key:
            key_name = f"{self.provider.upper()}_API_KEY"
            raise ValueError(
                f"Cloud API key missing for provider '{self.provider}'. "
                f"Please set the {key_name} environment variable or provide it in the UI sidebar."
            )

        images = []
        if image_paths:
            for p in image_paths:
                p_obj = Path(p)
                if p_obj.exists():
                    images.append(p_obj)

        if self.provider == "groq":
            return self._generate_groq(
                system_prompt=system_prompt,
                user_text=user_text,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif self.provider == "gemini":
            return self._generate_gemini(
                system_prompt=system_prompt,
                user_text=user_text,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            return self._generate_openai(
                system_prompt=system_prompt,
                user_text=user_text,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    def _generate_gemini(
        self,
        system_prompt: str,
        user_text: str,
        images: List[Path],
        max_tokens: int,
        temperature: float,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        contents: List[Any] = []
        # Attach raster images
        for img_path in images:
            try:
                pil_img = Image.open(img_path)
                contents.append(pil_img)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path} for Gemini: {e}")

        # Attach textual evidence and instructions
        contents.append(user_text)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max(max_tokens, 2048),
        )

        candidate_models = [self.model_name]
        for fallback in [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-2.0-flash-lite",
        ]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        import time

        last_error = None
        for m in candidate_models:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=contents,
                        config=config,
                    )
                    return (response.text or "").strip()
                except Exception as e:
                    last_error = e
                    err_msg = str(e).lower()
                    if "503" in err_msg or "high demand" in err_msg or "resource_exhausted" in err_msg or "429" in err_msg:
                        print(f"Notice: Gemini {m} attempt {attempt + 1} experienced transient spike: {e}. Retrying in 1.5s...")
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    else:
                        print(f"Notice: Gemini generation with {m} returned: {e}. Trying next candidate model...")
                        break

        raise RuntimeError(f"All candidate Gemini models failed. Last error: {last_error}")

    def _generate_openai(
        self,
        system_prompt: str,
        user_text: str,
        images: List[Path],
        max_tokens: int,
        temperature: float,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_text}
        ]

        for img_path in images:
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = img_path.suffix.lower().replace(".", "")
                mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp") else "image/jpeg"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                })
            except Exception as e:
                print(f"Warning: Failed to encode image {img_path} for OpenAI: {e}")

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _generate_groq(
        self,
        system_prompt: str,
        user_text: str,
        images: List[Path],
        max_tokens: int,
        temperature: float,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
        )

        content = user_text
        if images:
            image_notes = "\n".join([f"- Referenced figure / diagram: {img.name}" for img in images])
            content = f"{user_text}\n\n[Referenced Document Figures]:\n{image_notes}\n"

        candidate_models = [self.model_name]
        for fb in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
            if fb not in candidate_models:
                candidate_models.append(fb)

        last_error = None
        for m in candidate_models:
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_error = e
                print(f"Notice: Groq model {m} returned {e}. Trying fallback...")

        raise RuntimeError(f"All candidate Groq models failed. Last error: {last_error}")

