from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class ImageEmbeddingModel:
    """SigLIP 2 Vision and Text Embedding Model.
    
    Generates aligned image and text representations in a shared multimodal embedding space.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
    ):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pool_features(output) -> torch.Tensor:
        """Compatible with model outputs that expose pooled or hidden features."""
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
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors along the last dimension."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def get_dimension(self) -> int:
        """Return the output embedding dimension."""
        if hasattr(self.model.config, "vision_config") and hasattr(self.model.config.vision_config, "hidden_size"):
            return self.model.config.vision_config.hidden_size
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        return 768

    def encode_images(
        self,
        image_paths: list[str | Path],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Encode images into normalized float32 vectors."""
        if not image_paths:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        all_vectors = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(Path(p)).convert("RGB") for p in batch_paths]

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                if hasattr(self.model, "get_image_features"):
                    outputs = self.model.get_image_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                features = self._pool_features(outputs)

            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = np.asarray(features)

            all_vectors.append(features_np)

        combined = np.vstack(all_vectors)
        return self._normalize(combined)

    def encode_text(
        self,
        texts: list[str],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Encode text strings into the shared SigLIP multimodal vector space."""
        if not texts:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                if hasattr(self.model, "get_text_features"):
                    outputs = self.model.get_text_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                features = self._pool_features(outputs)

            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = np.asarray(features)

            all_vectors.append(features_np)

        combined = np.vstack(all_vectors)
        return self._normalize(combined)
