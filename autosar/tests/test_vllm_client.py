"""Test cases for vllm_client module"""
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm_client import VLLMClient


class TestVLLMClient(TestCase):
    """Test cases for VLLMClient"""

    @patch('vllm_client.OpenAI')
    def test_init_with_config(self, mock_openai):
        """Test initialization with config"""
        mock_config = MagicMock()
        mock_config.api_key = "test_key"
        mock_config.chat_base_url = "http://test:8000/v1"
        mock_config.embedding_base_url = "http://test:8001/v1"
        mock_config.chat_model = "test-model"
        mock_config.embedding_model = "test-embed"
        mock_config.get_chat_params.return_value = {}
        mock_config.get_embedding_params.return_value = {}

        client = VLLMClient(mock_config)

        self.assertEqual(client.config, mock_config)
        mock_openai.assert_called()

    @patch('vllm_client.OpenAI')
    def test_generate_with_llmresponse(self, mock_openai):
        """Test generate method returns LLMResponse"""
        mock_config = MagicMock()
        mock_config.api_key = "test_key"
        mock_config.chat_base_url = "http://test:8000/v1"
        mock_config.embedding_base_url = "http://test:8001/v1"
        mock_config.chat_model = "test-model"
        mock_config.embedding_model = "test-embed"
        mock_config.get_chat_params.return_value = {}

        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "test-model"

        mock_openai_instance.chat.completions.create.return_value = mock_response

        client = VLLMClient(mock_config)
        result = client.generate("Test prompt")

        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Test response")
        self.assertIn('metadata', result.__dict__)

    @patch('vllm_client.OpenAI')
    def test_embed(self, mock_openai):
        """Test embed method"""
        mock_config = MagicMock()
        mock_config.api_key = "test_key"
        mock_config.chat_base_url = "http://test:8000/v1"
        mock_config.embedding_base_url = "http://test:8001/v1"
        mock_config.chat_model = "test-model"
        mock_config.embedding_model = "test-embed"
        mock_config.get_embedding_params.return_value = {}

        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance

        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]

        mock_openai_instance.embeddings.create.return_value = mock_response

        client = VLLMClient(mock_config)
        result = client.embed("Test text")

        self.assertEqual(result, [0.1, 0.2, 0.3])

    @patch('vllm_client.OpenAI')
    def test_health_check(self, mock_openai):
        """Test health check method"""
        mock_config = MagicMock()
        mock_config.api_key = "test_key"
        mock_config.chat_base_url = "http://test:8000/v1"
        mock_config.embedding_base_url = "http://test:8001/v1"
        mock_config.chat_model = "test-model"
        mock_config.embedding_model = "test-embed"

        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_openai_instance.models.list.return_value = MagicMock()

        client = VLLMClient(mock_config)
        result = client.health_check()

        self.assertIn('chat_server', result)
        self.assertIn('embedding_server', result)
        self.assertIn('chat_url', result)
        self.assertIn('embed_url', result)

