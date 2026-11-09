import json
from unittest.mock import MagicMock
from src.context.context_organizer import ContextOrganizer
from src.context.context_builder import ContextBuilder
from src.context.image_describer import ImageDescriber


def test_context_organizer_page_grouping():
    organizer = ContextOrganizer()

    text_results = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "page_number": 2,
            "chunk_index": 1,
            "content": "Second paragraph on page 2.",
            "source_file": "doc1.pdf",
            "related_image_ids": ["img1"],
        },
        {
            "chunk_id": "c0",
            "document_id": "doc1",
            "page_number": 2,
            "chunk_index": 0,
            "content": "First paragraph on page 2.",
            "source_file": "doc1.pdf",
            "related_image_ids": [],
        },
        {
            "chunk_id": "c2",
            "document_id": "doc1",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Introduction on page 1.",
            "source_file": "doc1.pdf",
            "related_image_ids": [],
        },
    ]

    image_results = [
        {
            "image_id": "img1",
            "document_id": "doc1",
            "page_number": 2,
            "image_index": 0,
            "image_path": "images/img1.png",
            "width": 640,
            "height": 480,
            "surrounding_text": "Figure 1 diagram",
            "related_text_chunk_ids": ["c0", "c1"],
        }
    ]

    pages = organizer.organize(text_results, image_results)
    assert len(pages) == 2

    # First page should be page 1
    assert pages[0]["page_number"] == 1
    assert len(pages[0]["text_chunks"]) == 1
    assert len(pages[0]["images"]) == 0

    # Second page should be page 2
    assert pages[1]["page_number"] == 2
    assert len(pages[1]["text_chunks"]) == 2
    # Verify chunks are sorted by chunk_index (c0 then c1)
    assert pages[1]["text_chunks"][0]["chunk_id"] == "c0"
    assert pages[1]["text_chunks"][1]["chunk_id"] == "c1"

    # Verify image is grouped on page 2
    assert len(pages[1]["images"]) == 1
    assert pages[1]["images"][0]["image_id"] == "img1"
    assert pages[1]["images"][0]["related_text_chunk_ids"] == ["c0", "c1"]


def test_image_describer_cache(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({"test_img.png": "Cached description of diagram."}))

    describer = ImageDescriber(cache_path=str(cache_file))
    # Should fetch from cache without attempting to load Qwen model
    desc = describer.describe_image("path/to/test_img.png")
    assert desc == "Cached description of diagram."
    assert describer.model is None


def test_context_builder_without_descriptions():
    builder = ContextBuilder(enable_image_descriptions=False)

    text_results = [
        {
            "chunk_id": "c1",
            "document_id": "docA",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Deep learning explanation.",
            "source_file": "docA.pdf",
        }
    ]
    image_results = [
        {
            "image_id": "img1",
            "document_id": "docA",
            "page_number": 1,
            "image_path": "images/docA/img1.png",
        }
    ]

    context = builder.build(
        query="What is deep learning?",
        retrieval_results={"text_results": text_results, "image_results": image_results},
    )

    assert context["query"] == "What is deep learning?"
    assert len(context["pages"]) == 1
    page = context["pages"][0]
    assert page["document_id"] == "docA"
    assert len(page["text_chunks"]) == 1
    assert len(page["images"]) == 1
    assert page["images"][0]["description"] is None

    stats = context["statistics"]
    assert stats["text_chunks_retrieved"] == 1
    assert stats["images_retrieved"] == 1
    assert stats["images_described"] == 0
    assert stats["pages_with_evidence"] == 1


def test_context_builder_with_mock_describer():
    builder = ContextBuilder(enable_image_descriptions=True, max_images_to_describe=1)
    mock_describer = MagicMock()
    mock_describer.describe_image.return_value = "Mock description of neural diagram."
    builder.image_describer = mock_describer

    text_results = [
        {"chunk_id": "c1", "document_id": "docA", "page_number": 1, "content": "Text"}
    ]
    image_results = [
        {"image_id": "img1", "document_id": "docA", "page_number": 1, "image_path": "p1.png"},
        {"image_id": "img2", "document_id": "docA", "page_number": 2, "image_path": "p2.png"},
    ]

    context = builder.build(
        query="Explain architecture",
        retrieval_results={"text_results": text_results, "image_results": image_results},
    )

    stats = context["statistics"]
    assert stats["images_described"] == 1
    # Only 1 image should have description due to max_images_to_describe=1
    described_images = [
        img for p in context["pages"] for img in p["images"] if img.get("description") is not None
    ]
    assert len(described_images) == 1
    assert described_images[0]["description"] == "Mock description of neural diagram."

