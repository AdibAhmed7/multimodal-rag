from unittest.mock import MagicMock
from src.llm.prompt_builder import PromptBuilder
from src.llm.response_formatter import ResponseFormatter
from src.llm.answer_generator import AnswerGenerator
from src.llm.model_loader import QwenModel


def test_prompt_builder():
    builder = PromptBuilder()
    assert "STRICT RULES" in builder.SYSTEM_PROMPT

    pages = [
        {
            "document_id": "doc1",
            "page_number": 2,
            "text_chunks": [
                {"chunk_id": "c1", "content": "Transformer encoder architecture."}
            ],
            "images": [
                {"image_id": "img1", "description": "Encoder layer diagram."}
            ],
        }
    ]

    prompt = builder.build("Explain encoder", pages)
    assert "User Question: Explain encoder" in prompt
    assert "--- Document: doc1 | Page: 2 ---" in prompt
    assert "[Chunk c1]: Transformer encoder architecture." in prompt
    assert "[Image img1] Description: Encoder layer diagram." in prompt
    assert "[Page N]" in prompt


def test_prompt_builder_empty():
    builder = PromptBuilder()
    prompt = builder.build("Empty question", [])
    assert "No relevant document sections found." in prompt


def test_response_formatter():
    result = {
        "query": "What is an attention head?",
        "answer": "An attention head computes self-attention. [Page 2]",
        "sources": [
            {
                "type": "text",
                "id": "c1",
                "document_id": "example",
                "page_number": 2,
                "content": "Attention explanation",
            },
            {
                "type": "image",
                "id": "img1",
                "document_id": "example",
                "page_number": 2,
                "image_path": "img1.png",
            },
        ],
        "attached_images": [
            {
                "image_id": "img1",
                "document_id": "example",
                "page_number": 2,
                "image_path": "path/img1.png",
            }
        ],
    }

    formatted = ResponseFormatter.format(result)
    assert formatted["query"] == "What is an attention head?"
    assert formatted["answer"] == "An attention head computes self-attention. [Page 2]"
    assert formatted["model"] == "Qwen2.5-VL-3B-Instruct"
    assert "generated_at" in formatted

    # Sources
    sources = formatted["sources"]
    assert len(sources["pages"]) == 1
    assert sources["pages"][0]["page_number"] == 2
    assert len(sources["text_chunks"]) == 1
    assert len(sources["images"]) == 1

    # Display & stats
    assert len(formatted["display"]["images"]) == 1
    stats = formatted["statistics"]
    assert stats["attached_image_count"] == 1
    assert stats["text_source_count"] == 1
    assert stats["image_source_count"] == 1


def test_answer_generator_with_mock_model(tmp_path):
    mock_model = MagicMock()
    mock_model.generate.return_value = "The Transformer architecture relies on attention. [Page 2]"

    test_img = tmp_path / "test.png"
    test_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    generator = AnswerGenerator(model=mock_model, max_images=1)

    context = {
        "pages": [
            {
                "document_id": "doc1",
                "page_number": 2,
                "text_chunks": [
                    {"chunk_id": "c1", "content": "Transformer content.", "source_file": "doc1.pdf"}
                ],
                "images": [
                    {"image_id": "img1", "image_path": str(test_img), "description": "Diagram"}
                ],
            }
        ]
    }

    output = generator.generate("What is Transformer?", context)
    assert output["query"] == "What is Transformer?"
    assert output["answer"] == "The Transformer architecture relies on attention. [Page 2]"
    assert len(output["attached_images"]) == 1
    assert len(output["sources"]) == 2  # 1 text chunk + 1 image


def test_qwen_fallback_generation():
    model = QwenModel.__new__(QwenModel)
    model.is_fallback = True

    messages = [
        {
            "role": "user",
            "content": (
                "--- Document: example | Page: 2 ---\n"
                "[Chunk c1]: Self-attention mechanisms compute weights across tokens.\n"
                "[Image img1] Description: Diagram of self-attention.\n"
            ),
        }
    ]

    answer = model._fallback_generate(messages)
    assert "[Page 2]" in answer
    assert "Self-attention" in answer

