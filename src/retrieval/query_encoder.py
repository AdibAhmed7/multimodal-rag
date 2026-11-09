from pathlib import Path
from typing import Union
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer


class QueryEncoder:
    """Encodes user queries into vector spaces for text and image retrieval.

    - BGE-M3 (1024D) for text-to-text retrieval.
    - SigLIP 2 (768D) for text-to-image and image-to-image retrieval.
    """

    def __init__(
        self,
        text_model_name: str = "BAAI/bge-m3",
        image_model_name: str = "google/siglip2-base-patch16-224",
    ):
        print(f"Loading BGE-M3 text model ({text_model_name})...")
        try:
            self.text_model = SentenceTransformer(text_model_name, local_files_only=True)
        except Exception:
            self.text_model = SentenceTransformer(text_model_name)

        print(f"Loading SigLIP 2 multimodal model ({image_model_name})...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.image_processor = AutoProcessor.from_pretrained(image_model_name, local_files_only=True)
            self.image_model = AutoModel.from_pretrained(image_model_name, local_files_only=True)
        except Exception:
            self.image_processor = AutoProcessor.from_pretrained(image_model_name)
            self.image_model = AutoModel.from_pretrained(image_model_name)

        self.image_model.to(self.device)
        self.image_model.eval()

    @staticmethod
    def _pool_features(output) -> torch.Tensor:
        """Extract pooled tensor features from model output."""
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state.mean(dim=1)
        return output

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalize a 1D vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (vector / norm).astype(np.float32)

    def encode_text_for_text_search(self, query: str) -> np.ndarray:
        """Encode query text using BGE-M3 into a normalized 1024D vector."""
        if not query or not query.strip():
            return np.zeros(1024, dtype=np.float32)

        vector = self.text_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)

    def encode_text_for_image_search(self, query: str) -> np.ndarray:
        """Encode query text using SigLIP 2 into a normalized 768D vector."""
        if not query or not query.strip():
            return np.zeros(768, dtype=np.float32)

        inputs = self.image_processor(
            text=[query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            if hasattr(self.image_model, "get_text_features"):
                outputs = self.image_model.get_text_features(**inputs)
            else:
                outputs = self.image_model(**inputs)
            features = self._pool_features(outputs)

        if isinstance(features, torch.Tensor):
            vector = features.cpu().numpy()[0]
        else:
            vector = np.asarray(features)[0]

        return self._normalize(vector)

    def encode_image(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """Encode an image using SigLIP 2 into a normalized 768D vector."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(Path(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        inputs = self.image_processor(images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            if hasattr(self.image_model, "get_image_features"):
                outputs = self.image_model.get_image_features(**inputs)
            else:
                outputs = self.image_model(**inputs)
            features = self._pool_features(outputs)

        if isinstance(features, torch.Tensor):
            vector = features.cpu().numpy()[0]
        else:
            vector = np.asarray(features)[0]

        return self._normalize(vector)
