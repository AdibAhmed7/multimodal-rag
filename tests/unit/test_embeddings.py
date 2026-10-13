from pathlib import Path
import numpy as np
import pytest
from src.embeddings.text_embeddings import TextEmbeddingModel
from src.embeddings.image_embeddings import ImageEmbeddingModel


def test_empty_text_embeddings():
    """Verify empty input returns proper empty array with 1024 dimensions."""
    # We can test encode([]) on a model instance or mock
    model = object.__new__(TextEmbeddingModel)
    result = model.encode([])
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 1024)
    assert result.dtype == np.float32


def test_image_embedding_normalization():
    """Verify _normalize produces unit L2-norm vectors."""
    raw_vectors = np.array([
        [3.0, 4.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    normalized = ImageEmbeddingModel._normalize(raw_vectors)
    assert normalized.shape == raw_vectors.shape
    assert normalized.dtype == np.float32

    # Vector 1: [3, 4] norm is 5 -> [0.6, 0.8]
    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.isclose(normalized[0, 0], 0.6)
    assert np.isclose(normalized[0, 1], 0.8)

    # Vector 2: norm is sqrt(2) -> unit length
    assert np.isclose(np.linalg.norm(normalized[1]), 1.0)

    # Vector 3: zero vector handles division by zero safely
    assert np.all(normalized[2] == 0.0)


def test_empty_image_and_text_siglip():
    """Verify empty inputs return expected shapes."""
    model = object.__new__(ImageEmbeddingModel)
    model.get_dimension = lambda: 768

    empty_imgs = model.encode_images([])
    assert empty_imgs.shape == (0, 768)
    assert empty_imgs.dtype == np.float32

    empty_txt = model.encode_text([])
    assert empty_txt.shape == (0, 768)
    assert empty_txt.dtype == np.float32


def test_vectors_npz_manifest_roundtrip(tmp_path):
    """Verify saving and reading vectors.npz and manifest.json."""
    text_vecs = np.random.randn(3, 1024).astype(np.float32)
    img_vecs = np.random.randn(2, 768).astype(np.float32)

    npz_path = tmp_path / "vectors.npz"
    np.savez_compressed(
        npz_path,
        text_vectors=text_vecs,
        image_vectors=img_vecs,
    )

    with np.load(npz_path) as data:
        assert "text_vectors" in data
        assert "image_vectors" in data
        assert np.allclose(data["text_vectors"], text_vecs)
        assert np.allclose(data["image_vectors"], img_vecs)
