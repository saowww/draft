"""Test cases for edge_extractor module"""
import json
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.graph_extractor.schema import ExtractedEdges
from edge_extractor import AUTOSAREdgeExtractor


class TestAUTOSAREdgeExtractor(TestCase):
    """Test cases for AUTOSAREdgeExtractor"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = MagicMock()
        self.extractor = AUTOSAREdgeExtractor(self.mock_client)

    def test_extract_empty_text(self):
        """Test extracting from empty text"""
        result = self.extractor.extract("", [], "test.json")
        self.assertEqual(len(result.edges), 0)

    def test_extract_no_nodes(self):
        """Test extracting with no nodes"""
        result = self.extractor.extract("Some text", [], "test.json")
        self.assertEqual(len(result.edges), 0)

    def test_extract_with_llmresponse(self):
        """Test extracting with LLMResponse object"""
        mock_response = MagicMock()
        mock_response.message = json.dumps({
            "edges": [
                {
                    "source": "Node1",
                    "relation": "relates_to",
                    "target": "Node2"
                }
            ]
        })

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        nodes = [
            {"name": "Node1", "semantic_type": "Type1"},
            {"name": "Node2", "semantic_type": "Type2"}
        ]
        result = self.extractor.extract(text, nodes, "test.json")

        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0].source, "Node1")
        self.assertEqual(result.edges[0].relation, "relates_to")
        self.assertEqual(result.edges[0].target, "Node2")

    def test_extract_with_choices_response(self):
        """Test extracting with OpenAI-compatible response"""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "edges": [{"source": "N1", "relation": "r1", "target": "N2"}]
        })
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        nodes = [{"name": "N1", "semantic_type": "T1"}]
        result = self.extractor.extract(text, nodes, "test.json")

        self.assertEqual(len(result.edges), 1)

    def test_extract_with_string_response(self):
        """Test extracting with string response"""
        json_str = json.dumps({
            "edges": [{"source": "N1", "relation": "r1", "target": "N2"}]
        })
        self.mock_client.generate.return_value = json_str

        text = "Test text"
        nodes = [{"name": "N1", "semantic_type": "T1"}]
        result = self.extractor.extract(text, nodes, "test.json")

        self.assertEqual(len(result.edges), 1)

    def test_extract_with_markdown_fences(self):
        """Test extracting with markdown code fences"""
        json_content = json.dumps({
            "edges": [{"source": "N1", "relation": "r1", "target": "N2"}]
        })
        mock_response = MagicMock()
        mock_response.message = f"```json\n{json_content}\n```"

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        nodes = [{"name": "N1", "semantic_type": "T1"}]
        result = self.extractor.extract(text, nodes, "test.json")

        self.assertEqual(len(result.edges), 1)

    def test_extract_with_invalid_json(self):
        """Test extracting with invalid JSON"""
        mock_response = MagicMock()
        mock_response.message = "This is not valid JSON"

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        nodes = [{"name": "N1", "semantic_type": "T1"}]
        
        with self.assertRaises(ValueError):
            self.extractor.extract(text, nodes, "test.json")

    def test_extract_source_file_added(self):
        """Test that source_file is added to edges"""
        mock_response = MagicMock()
        mock_response.message = json.dumps({
            "edges": [
                {"source": "N1", "relation": "r1", "target": "N2"}
            ]
        })

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        nodes = [{"name": "N1", "semantic_type": "T1"}]
        result = self.extractor.extract(text, nodes, "test_file.json")

        # Check that source_file is in edge dict
        edge_dict = result.edges[0].dict()
        self.assertEqual(edge_dict.get('source_file'), "test_file.json")

    def test_extract_multiple_edges(self):
        """Test extracting multiple edges"""
        mock_response = MagicMock()
        mock_response.message = json.dumps({
            "edges": [
                {"source": "N1", "relation": "r1", "target": "N2"},
                {"source": "N2", "relation": "r2", "target": "N3"}
            ]
        })

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        nodes = [
            {"name": "N1", "semantic_type": "T1"},
            {"name": "N2", "semantic_type": "T2"},
            {"name": "N3", "semantic_type": "T3"}
        ]
        result = self.extractor.extract(text, nodes, "test.json")

        self.assertEqual(len(result.edges), 2)

    def test_extract_empty_edges_response(self):
        """Test extracting with empty edges in response"""
        mock_response = MagicMock()
        mock_response.message = json.dumps({"edges": []})

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        nodes = [{"name": "N1", "semantic_type": "T1"}]
        result = self.extractor.extract(text, nodes, "test.json")

        self.assertEqual(len(result.edges), 0)

    def test_extract_exception_handling(self):
        """Test exception handling during extraction"""
        self.mock_client.generate.side_effect = Exception("Test error")

        text = "Test text"
        nodes = [{"name": "N1", "semantic_type": "T1"}]
        result = self.extractor.extract(text, nodes, "test.json")

        self.assertEqual(len(result.edges), 0)

    def test_extract_prompt_entities_format(self):
        """Test that prompt entities are formatted correctly"""
        mock_response = MagicMock()
        mock_response.message = json.dumps({"edges": []})

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        nodes = [
            {"name": "Node1", "semantic_type": "Type1"},
            {"name": "Node2", "semantic_type": "Type2"}
        ]
        self.extractor.extract(text, nodes, "test.json")

        # Check that generate was called
        self.mock_client.generate.assert_called_once()
        call_args = self.mock_client.generate.call_args
        prompt = call_args[1]['user_prompt']
        
        # Check that entities are in the prompt
        self.assertIn("Node1", prompt)
        self.assertIn("Type1", prompt)
        self.assertIn("Node2", prompt)
        self.assertIn("Type2", prompt)

