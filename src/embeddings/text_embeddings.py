import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import numpy as np
from google import genai


class TextEmbeddingModel:
    """Gemini Cloud Text Embedding Model.

    Encodes text chunks into dense 3072-dimensional normalized vectors
    using Google Gemini API (gemini-embedding-001).
    """

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        dimension: int = 3072,
    ):
        self.model_name = model_name
        self.dimension = dimension

        if not api_key:
            # Look for .env in multimodal-rag root or current working dir
            for env_candidate in [
                Path(__file__).resolve().parent.parent.parent / ".env",
                Path.cwd() / ".env",
                Path.cwd() / "multimodal-rag" / ".env",
            ]:
                if env_candidate.exists():
                    load_dotenv(env_candidate)
                    break
            api_key = os.environ.get("GEMINI_API_KEY")

        self.api_key = api_key
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Encode a list of text strings into normalized float32 vectors using Gemini."""
        dim = getattr(self, "dimension", 3072)
        if not texts:
            return np.empty((0, dim), dtype=np.float32)

        if self.client is None:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY to your environment or .env file."
            )

        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
            )
            for emb in response.embeddings:
                all_vectors.append(emb.values)

        arr = np.array(all_vectors, dtype=np.float32)

        # L2 Normalization for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = arr / norms
        return normalized.astype(np.float32)
