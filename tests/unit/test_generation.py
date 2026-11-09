"""Unit tests for Phase 8 generation modules."""

import tempfile
from pathlib import Path
from PIL import Image
import pytest

from src.generation.request_router import RequestRouter
from src.generation.image_prompt_builder import ImagePromptBuilder
from src.generation.image_generator import ImageGenerator
from src.generation.visual_response_builder import VisualResponseBuilder


class TestRequestRouter:
    """Tests for intent routing logic."""

    def setup_method(self):
        self.router = RequestRouter()

    def test_route_answer(self):
        result = self.router.route("Explain the main concept of this PDF")
        assert result["intent"] == "answer"
        assert result["retrieve_pdf_images"] is True
        assert result["generate_new_image"] is False
        assert result["answer_question"] is True

    def test_route_retrieve_pdf_images(self):
        result = self.router.route("Show me the diagram from the PDF")
        assert result["intent"] == "retrieve_pdf_images"
        assert result["retrieve_pdf_images"] is True
        assert result["generate_new_image"] is False
        assert result["answer_question"] is False

    def test_route_generate_image(self):
        result = self.router.route("Generate a new neural network illustration")
        assert result["intent"] == "generate_image"
        assert result["retrieve_pdf_images"] is False
        assert result["generate_new_image"] is True
        assert result["answer_question"] is False

    def test_route_answer_and_generate(self):
        result = self.router.route("Generate a new diagram and show the original PDF diagram")
        assert result["intent"] == "answer_and_generate"
        assert result["retrieve_pdf_images"] is True
        assert result["generate_new_image"] is True
        assert result["answer_question"] is True

    def test_case_insensitivity_and_whitespace(self):
        result = self.router.route("  GENERATE A PICTURE of attention mechanism  ")
        assert result["intent"] == "generate_image"
        assert result["generate_new_image"] is True


class TestImagePromptBuilder:
    """Tests for prompt builder expansion."""

    def setup_method(self):
        self.builder = ImagePromptBuilder()

    def test_build_basic(self):
        prompt = self.builder.build("A transformer encoder architecture")
        assert "A transformer encoder architecture" in prompt
        assert "Clear composition." in prompt
        assert "Visual requirements:" in prompt

    def test_build_with_context(self):
        context = "The architecture contains 6 encoder layers with self-attention."
        prompt = self.builder.build("Transformer model", context=context)
        assert "Transformer model" in prompt
        assert "Additional information from the PDF:" in prompt
        assert context in prompt


class TestImageGenerator:
    """Tests for image generator lifecycle and fallback generation."""

    def test_generate_fallback_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = ImageGenerator(output_dir=tmp_dir, allow_fallback=True)
            # Force fallback mode to verify synthetic diagram pipeline
            generator.is_fallback = True

            meta = generator.generate(
                prompt="Neural Network Schema",
                num_inference_steps=2,
                seed=123,
                width=512,
                height=512,
            )

            assert Path(meta["image_path"]).exists()
            assert meta["image_type"] == "ai_generated"
            assert meta["width"] == 512
            assert meta["height"] == 512

            # Verify image file is a valid 512x512 PNG
            with Image.open(meta["image_path"]) as img:
                assert img.size == (512, 512)
                assert img.format == "PNG"

            # Test memory unloading
            generator.unload()
            assert generator.pipeline is None


class TestVisualResponseBuilder:
    """Tests for response aggregation and retrieved vs generated image separation."""

    def setup_method(self):
        self.builder = VisualResponseBuilder()

    def test_build_response_separation(self):
        retrieved = [
            {"image_id": "r1", "image_path": "data/images/doc_p1.png", "page_number": 1}
        ]
        generated = [
            {"image_id": "g1", "image_path": "data/generated/images/g1.png", "model": "sd-turbo"}
        ]
        sources = [{"type": "text", "page_number": 1}]

        response = self.builder.build(
            query="Compare original diagram with new illustration",
            intent="answer_and_generate",
            answer="Here is the explanation comparing both.",
            retrieved_images=retrieved,
            generated_images=generated,
            sources=sources,
        )

        assert response["query"] == "Compare original diagram with new illustration"
        assert response["intent"] == "answer_and_generate"
        assert response["answer"] == "Here is the explanation comparing both."
        assert len(response["retrieved_images"]) == 1
        assert response["retrieved_images"][0]["image_id"] == "r1"
        assert len(response["generated_images"]) == 1
        assert response["generated_images"][0]["image_id"] == "g1"
        assert response["statistics"]["retrieved_image_count"] == 1
        assert response["statistics"]["generated_image_count"] == 1
        assert response["statistics"]["source_count"] == 1
        assert "generated_at" in response
