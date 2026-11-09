from src.context.context_organizer import ContextOrganizer


def test_context_organizer():
    organizer = ContextOrganizer()
    text_results = [
        {
            "document_id": "example",
            "page_number": 1,
            "chunk_id": "chunk_1",
            "chunk_index": 0,
            "content": "Introduction",
        }
    ]
    image_results = [
        {
            "document_id": "example",
            "page_number": 1,
            "image_id": "image_1",
            "related_text_chunk_ids": ["chunk_1"],
        }
    ]
    result = organizer.organize(
        text_results,
        image_results,
    )
    assert len(result) == 1
    assert result[0]["document_id"] == "example"
    assert len(result[0]["text_chunks"]) == 1

