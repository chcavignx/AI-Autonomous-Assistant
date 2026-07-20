#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

from examples.VAD.voice_agent_offline import ResponsesConfig, SimpleVoiceAgent
from src.utils.config import Config


class TestVoiceAgentResponses(unittest.TestCase):
    @patch("examples.VAD.voice_agent_offline.ASREngine")
    @patch("examples.VAD.voice_agent_offline.TTSEngine")
    @patch("examples.VAD.voice_agent_offline.WakeWordDetector")
    def test_english_responses(self, mock_wake, mock_tts, mock_asr):
        # Create a config with language "en"
        config_data = {
            "responses": {"language": "en", "files": {"en": "data/responses_en.yaml", "fr": "data/responses_fr.yaml"}},
            "wake": {"wake_word": "hello_jarvis"},
            "llm": {"url": "http://localhost:11434/api/generate", "model": "llama3.2:latest", "timeout": 1.0},
        }
        config = Config(**config_data)
        responses_config = ResponsesConfig.model_validate(config_data["responses"])

        with patch("src.utils.config.load_config", return_value=config):
            agent = SimpleVoiceAgent(responses_config=responses_config)

            # Test hello/hi
            resp, should_cont = agent._generate_response("hello assistant")
            assert "jarvis" in resp
            assert should_cont

            # Test action time
            resp, should_cont = agent._generate_response("what time is it?")
            assert "The current time is" in resp
            assert should_cont

            # Test exit/stop
            resp, should_cont = agent._generate_response("please stop now")
            assert "Goodbye" in resp
            assert not should_cont

    @patch("examples.VAD.voice_agent_offline.ASREngine")
    @patch("examples.VAD.voice_agent_offline.TTSEngine")
    @patch("examples.VAD.voice_agent_offline.WakeWordDetector")
    def test_french_responses(self, mock_wake, mock_tts, mock_asr):
        # Create a config with language "fr"
        config_data = {
            "responses": {"language": "fr", "files": {"en": "data/responses_en.yaml", "fr": "data/responses_fr.yaml"}},
            "wake": {"wake_word": "bonjour_jarvis"},
            "llm": {"url": "http://localhost:11434/api/generate", "model": "llama3.2:latest", "timeout": 1.0},
        }
        config = Config(**config_data)
        responses_config = ResponsesConfig.model_validate(config_data["responses"])

        with patch("src.utils.config.load_config", return_value=config):
            agent = SimpleVoiceAgent(responses_config=responses_config)

            # Test hello/hi (French: bonjour)
            resp, should_cont = agent._generate_response("bonjour assistant")
            assert "jarvis" in resp
            assert should_cont

            # Test exit/stop (French: arrête)
            resp, should_cont = agent._generate_response("arrête la musique")
            assert "Au revoir" in resp
            assert not should_cont

    def test_language_fallback_to_config(self):
        # Create a mock config and patch the global config in examples.VAD.voice_agent_offline
        mock_config = MagicMock()
        mock_config.asr.language = "fr"

        with patch("examples.VAD.voice_agent_offline.config", mock_config):
            # Instantiate ResponsesConfig without specifying language
            responses_config = ResponsesConfig()
            assert responses_config.language == "fr"

    @patch("requests.post")
    def test_generate_llm_response_payload(self, mock_post):
        # Mock success response from LLM
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello from mock LLM!"}
        mock_post.return_value = mock_response

        # Configure custom llm_payload
        custom_payload = {
            "model": "custom-model",
            "prompt": "Say hello to: {user_input} in model {model}",
            "stream": False,
            "temperature": 0.7,
        }

        responses_config = ResponsesConfig(
            language="en", files={"en": "data/responses_en.yaml"}, llm_payload=custom_payload
        )

        with (
            patch("examples.VAD.voice_agent_offline.ASREngine"),
            patch("examples.VAD.voice_agent_offline.TTSEngine"),
            patch("examples.VAD.voice_agent_offline.WakeWordDetector"),
        ):
            agent = SimpleVoiceAgent(responses_config=responses_config)
            # Query LLM response
            resp = agent._generate_llm_response(user_input="Alice", fallback_template="Fallback")

            # Assert response matches mock output
            assert resp == "Hello from mock LLM!"

            # Assert payload was correctly formatted
            mock_post.assert_called_once()
            _called_args, called_kwargs = mock_post.call_args
            called_json = called_kwargs.get("json", {})

            assert called_json.get("model") == "custom-model"
            assert called_json.get("prompt") == "Say hello to: Alice in model llama3.2:latest"
            assert not called_json.get("stream")
            assert called_json.get("temperature") == 0.7

    @patch("requests.post")
    def test_generate_response_fallback_no_llm(self, mock_post):
        # Create configuration with use_llm = False
        config_data = {
            "responses": {"language": "en", "use_llm": False, "files": {"en": "data/responses_en.yaml"}},
            "wake": {"wake_word": "hello_jarvis"},
        }
        config = Config(**config_data)
        responses_config = ResponsesConfig.model_validate(config_data["responses"])

        with (
            patch("src.utils.config.load_config", return_value=config),
            patch("examples.VAD.voice_agent_offline.ASREngine"),
            patch("examples.VAD.voice_agent_offline.TTSEngine"),
            patch("examples.VAD.voice_agent_offline.WakeWordDetector"),
        ):
            agent = SimpleVoiceAgent(responses_config=responses_config)

            # Input that does not match any keyword (should trigger fallback)
            resp, should_cont = agent._generate_response("some unrecognized query")

            # Check that requests.post was NOT called
            mock_post.assert_not_called()

            # Check that fallback message is formatted as plain text
            assert "You said: some unrecognized query. I'm still learning" in resp
            assert should_cont

    @patch("requests.post")
    def test_generate_response_fallback_with_llm(self, mock_post):
        # Mock success response from LLM
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello from LLM!"}
        mock_post.return_value = mock_response

        # Create configuration with use_llm = True
        config_data = {
            "responses": {"language": "en", "use_llm": True, "files": {"en": "data/responses_en.yaml"}},
            "wake": {"wake_word": "hello_jarvis"},
            "llm": {"url": "http://localhost:11434/api/generate", "model": "llama3.2:latest", "timeout": 1.0},
        }
        config = Config(**config_data)
        responses_config = ResponsesConfig.model_validate(config_data["responses"])

        with (
            patch("src.utils.config.load_config", return_value=config),
            patch("examples.VAD.voice_agent_offline.ASREngine"),
            patch("examples.VAD.voice_agent_offline.TTSEngine"),
            patch("examples.VAD.voice_agent_offline.WakeWordDetector"),
        ):
            agent = SimpleVoiceAgent(responses_config=responses_config)

            # Input that does not match any keyword (should trigger fallback)
            resp, should_cont = agent._generate_response("some unrecognized query")

            # Check that requests.post WAS called
            mock_post.assert_called_once()

            # Check that fallback message is generated by LLM
            assert resp == "Hello from LLM!"
            assert should_cont

    @patch("requests.post")
    def test_generate_llm_response_openai_payload(self, mock_post):
        # Mock success response from OpenAI-compatible API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello from mock OpenAI!"}}]}
        mock_post.return_value = mock_response

        # Configure custom llm_payload
        custom_payload = {"prompt": "Say hello to: {user_input} in model {model}"}

        responses_config = ResponsesConfig(
            language="en", files={"en": "data/responses_en.yaml"}, llm_payload=custom_payload
        )

        config_data = {
            "responses": {"language": "en"},
            "wake": {"wake_word": "hello_jarvis"},
            "llm": {
                "api_type": "openai",
                "url": "https://api.openai.com/v1/chat/completions",
                "model": "gpt-4o",
                "timeout": 2.0,
                "api_key": "sk-12345",
            },
        }
        config = Config(**config_data)

        with (
            patch("src.utils.config.load_config", return_value=config),
            patch("examples.VAD.voice_agent_offline.ASREngine"),
            patch("examples.VAD.voice_agent_offline.TTSEngine"),
            patch("examples.VAD.voice_agent_offline.WakeWordDetector"),
        ):
            agent = SimpleVoiceAgent(responses_config=responses_config)
            agent.config = config

            # Query LLM response
            resp = agent._generate_llm_response(user_input="Alice", fallback_template="Fallback")

            # Assert response matches mock output
            assert resp == "Hello from mock OpenAI!"

            # Assert payload was correctly formatted for OpenAI API
            mock_post.assert_called_once()
            _called_args, called_kwargs = mock_post.call_args
            called_json = called_kwargs.get("json", {})
            called_headers = called_kwargs.get("headers", {})

            assert called_json.get("model") == "gpt-4o"
            # OpenAI format uses messages structure
            assert called_json.get("messages") == [{"role": "user", "content": "Say hello to: Alice in model gpt-4o"}]
            assert not called_json.get("stream")
            assert called_headers.get("Authorization") == "Bearer sk-12345"

    @patch("requests.post")
    def test_generate_llm_response_direct_client_openai(self, mock_post):
        from src.llm.llm_client import generate_llm_response

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Direct response from mock OpenAI"}}]}
        mock_post.return_value = mock_response

        res = generate_llm_response(
            api_type="openai",
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4o",
            prompt="Hello there",
            timeout=2.0,
            api_key="sk-openai-key",
        )

        assert res == "Direct response from mock OpenAI"
        mock_post.assert_called_once()
        _called_args, called_kwargs = mock_post.call_args
        called_json = called_kwargs.get("json", {})
        called_headers = called_kwargs.get("headers", {})

        assert called_json["model"] == "gpt-4o"
        assert called_json["messages"] == [{"role": "user", "content": "Hello there"}]
        assert called_headers["Authorization"] == "Bearer sk-openai-key"
