import pytest
from pydantic import ValidationError

from src.chunking.chunk_models import ImageRecord, TextChunk
from src.chunking.image_chunker import ImageChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.text_chunker import TextChunker


def test_text_chunk_model():
    chunk = TextChunk(
        chunk_id="doc1_p001_c000",
        document_id="doc1",
        page_number=1,
        chunk_index=0,
        content="Sample content",
        source_file="doc1.pdf",
    )
    assert chunk.chunk_id == "doc1_p001_c000"
    assert chunk.related_image_ids == []

    # Content cannot be empty
    with pytest.raises(ValidationError):
        TextChunk(
            chunk_id="doc1_p001_c000",
            document_id="doc1",
            page_number=1,
            chunk_index=0,
            content="",
            source_file="doc1.pdf",
        )


def test_image_record_model():
    record = ImageRecord(
        image_id="img_001",
        document_id="doc1",
        page_number=1,
        image_index=1,
        image_path="path/to/img.png",
        extension="png",
        width=200,
        height=100,
    )
    assert record.image_id == "img_001"
    assert record.page_number == 1
    assert record.related_text_chunk_ids == []


def test_text_chunker_basic():
    chunker = TextChunker(chunk_size=100, overlap=20)
    doc = {
        "document_id": "test_doc",
        "filename": "test.pdf",
        "pages": [
            {"page_number": 1, "text": "Short page text."},
            {"page_number": 2, "text": ""},  # empty page should be skipped
            {
                "page_number": 3,
                "text": "First sentence. " * 10,  # ~160 chars, splits into chunks
            },
        ],
    }
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == "test_doc_p001_c000"
    assert chunks[0].content == "Short page text."


def test_image_chunker_linking():
    text_chunker = TextChunker()
    image_chunker = ImageChunker()

    doc = {
        "document_id": "test_doc",
        "filename": "test.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "Page one text content.",
                "images": [
                    {
                        "image_id": "img_p1_01",
                        "path": "path/img1.png",
                        "extension": "png",
                        "width": 100,
                        "height": 100,
                    }
                ],
            },
            {
                "page_number": 2,
                "text": "Page two text without images.",
                "images": [],
            },
        ],
    }

    text_chunks = text_chunker.chunk_document(doc)
    image_records = image_chunker.chunk_document(doc, text_chunks)

    assert len(image_records) == 1
    img = image_records[0]
    assert img.image_id == "img_p1_01"
    assert img.page_number == 1
    assert img.surrounding_text == "Page one text content."
    assert "test_doc_p001_c000" in img.related_text_chunk_ids

    # Text chunk on page 1 should have image linked
    assert "img_p1_01" in text_chunks[0].related_image_ids


def test_semantic_chunker_pipeline():
    pipeline = SemanticChunker()
    doc = {
        "document_id": "test_doc",
        "filename": "test.pdf",
        "page_count": 2,
        "pages": [
            {
                "page_number": 1,
                "text": "Page one text.",
                "images": [
                    {
                        "image_id": "img_1",
                        "path": "path/to/img1.png",
                        "extension": "png",
                    }
                ],
            },
            {
                "page_number": 2,
                "text": "Page two text.",
                "images": [],
            },
        ],
    }

    result = pipeline.process_document(doc)
    assert result["document_id"] == "test_doc"
    assert result["filename"] == "test.pdf"
    assert result["page_count"] == 2
    assert len(result["text_chunks"]) == 2
    assert len(result["image_records"]) == 1
