from unittest.mock import MagicMock, patch
import pytest
from src.llm.cloud_client import CloudMultimodalClient
from src.llm.answer_generator import AnswerGenerator


def test_cloud_client_initialization():
    client_gemini = CloudMultimodalClient(provider="gemini", api_key="test_key_gemini")
    assert client_gemini.provider == "gemini"
    assert client_gemini.model_name == "gemini-2.5-flash"
    assert client_gemini.api_key == "test_key_gemini"

    client_openai = CloudMultimodalClient(provider="openai", api_key="test_key_openai")
    assert client_openai.provider == "openai"
    assert client_openai.model_name == "gpt-4o-mini"
    assert client_openai.api_key == "test_key_openai"

    with pytest.raises(ValueError):
        CloudMultimodalClient(provider="unsupported_provider", api_key="key")


def test_cloud_client_missing_key_raises():
    client = CloudMultimodalClient(provider="gemini", api_key="")
    with pytest.raises(ValueError, match="Cloud API key missing"):
        client.generate(system_prompt="sys", user_text="hello")


def test_cloud_client_mock_gemini():
    client = CloudMultimodalClient(provider="gemini", api_key="fake_key")

    mock_response = MagicMock()
    mock_response.text = "This is a cloud-generated response citing [Page 5]."

    with patch("google.genai.Client") as mock_genai_client_class:
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_genai_client_class.return_value = mock_client_instance

        result = client.generate(
            system_prompt="System prompt instructions",
            user_text="User question about diagrams",
            image_paths=[],
        )

        assert result == "This is a cloud-generated response citing [Page 5]."
        mock_client_instance.models.generate_content.assert_called_once()


def test_cloud_client_mock_openai():
    client = CloudMultimodalClient(provider="openai", api_key="fake_key")

    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI answer citing [Page 2]."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client_instance

        result = client.generate(
            system_prompt="System instructions",
            user_text="What is on page 2?",
            image_paths=[],
        )

        assert result == "OpenAI answer citing [Page 2]."
        mock_client_instance.chat.completions.create.assert_called_once()


def test_answer_generator_online_mode():
    answer_gen = AnswerGenerator(
        online_mode=True,
        cloud_provider="gemini",
        api_key="fake_key",
    )
    assert answer_gen.online_mode is True
    assert answer_gen.cloud_client is not None
