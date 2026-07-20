import unittest
from unittest.mock import MagicMock, patch

from src.llm.llm_client import generate_llm_response


class TestLLMClient(unittest.TestCase):
    @patch("requests.post")
    def test_ollama_success(self, mock_post):
        # Mock success response for Ollama
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello from Ollama!"}
        mock_post.return_value = mock_response

        res = generate_llm_response(
            api_type="ollama", url="http://localhost:11434/api/generate", model="llama3.2", prompt="Hello", timeout=5.0
        )

        assert res == "Hello from Ollama!"
        mock_post.assert_called_once()
        _args, kwargs = mock_post.call_args
        assert kwargs["json"] == {"model": "llama3.2", "prompt": "Hello", "stream": False}
        assert kwargs["timeout"] == 5.0

    @patch("requests.post")
    def test_openai_success(self, mock_post):
        # Mock success response for OpenAI
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello from OpenAI!"}}]}
        mock_post.return_value = mock_response

        res = generate_llm_response(
            api_type="openai",
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4o",
            prompt="Hello",
            timeout=10.0,
            api_key="sk-test-key",
        )

        assert res == "Hello from OpenAI!"
        mock_post.assert_called_once()
        _args, kwargs = mock_post.call_args
        assert kwargs["json"] == {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"
        assert kwargs["timeout"] == 10.0

    @patch("requests.post")
    def test_payload_override(self, mock_post):
        # Mock success response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Overridden!"}
        mock_post.return_value = mock_response

        custom_payload = {"custom_field": "val"}
        res = generate_llm_response(
            api_type="ollama",
            url="http://localhost:11434/api/generate",
            model="llama3.2",
            prompt="Hello",
            payload_override=custom_payload,
        )

        assert res == "Overridden!"
        mock_post.assert_called_once()
        _args, kwargs = mock_post.call_args
        assert kwargs["json"] == custom_payload

    @patch("requests.post")
    def test_non_200_status(self, mock_post):
        # Mock status 500
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        res = generate_llm_response(
            api_type="ollama", url="http://localhost:11434/api/generate", model="llama3.2", prompt="Hello"
        )

        assert res is None

    @patch("requests.post")
    def test_request_exception(self, mock_post):
        # Mock exception
        mock_post.side_effect = Exception("Connection refused")

        res = generate_llm_response(
            api_type="ollama", url="http://localhost:11434/api/generate", model="llama3.2", prompt="Hello"
        )

        assert res is None
