from sentence_transformers import SentenceTransformer
import numpy as np


class TextEmbeddingModel:
    """BGE-M3 Dense Text Embedding Model.
    
    Encodes text chunks into dense 1024-dimensional normalized vectors.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Encode a list of text strings into normalized float32 vectors."""
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.astype(np.float32)
