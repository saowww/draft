"""Test cases for node_extractor module"""
import json
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from node_extractor import AUTOSARNodeExtractor


class TestAUTOSARNodeExtractor(TestCase):
    """Test cases for AUTOSARNodeExtractor"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = MagicMock()
        self.extractor = AUTOSARNodeExtractor(self.mock_client)

    def test_extract_context_normal_case(self):
        """Test extracting context around mention"""
        text = "This is a test sentence with mention in the middle."
        mention = "mention"
        left, right = self.extractor.extract_context(text, mention)
        self.assertIn("sentence with", left)
        self.assertIn("in the middle", right)

    def test_extract_context_mention_not_found(self):
        """Test extracting context when mention not found"""
        text = "This is a test sentence."
        mention = "notfound"
        left, right = self.extractor.extract_context(text, mention)
        self.assertEqual(left, "")
        self.assertEqual(right, "")

    def test_extract_context_empty_mention(self):
        """Test extracting context with empty mention"""
        text = "This is a test sentence."
        mention = ""
        left, right = self.extractor.extract_context(text, mention)
        self.assertEqual(left, "")
        self.assertEqual(right, "")

    def test_extract_empty_text(self):
        """Test extracting from empty text"""
        result = self.extractor.extract("", "test.json")
        self.assertEqual(result, [])

    def test_extract_whitespace_only(self):
        """Test extracting from whitespace-only text"""
        result = self.extractor.extract("   \n\t  ", "test.json")
        self.assertEqual(result, [])

    def test_extract_with_llmresponse(self):
        """Test extracting with LLMResponse object"""
        # Mock LLMResponse object
        mock_response = MagicMock()
        mock_response.message = json.dumps({
            "entities": [
                {
                    "name": "TestEntity",
                    "semantic_type": "Entity",
                    "mention": "TestEntity"
                }
            ]
        })

        self.mock_client.generate.return_value = mock_response

        text = "This is a test with TestEntity mentioned."
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], "TestEntity")
        self.assertEqual(result[0]['semantic_type'], "Entity")
        self.assertIn('context_left', result[0])
        self.assertIn('context_right', result[0])

    def test_extract_with_choices_response(self):
        """Test extracting with OpenAI-compatible response"""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "entities": [{"name": "Entity1", "semantic_type": "Type1"}]
        })
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], "Entity1")

    def test_extract_with_string_response(self):
        """Test extracting with string response"""
        json_str = json.dumps({
            "entities": [{"name": "Entity1", "semantic_type": "Type1"}]
        })
        self.mock_client.generate.return_value = json_str

        text = "Test text"
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(len(result), 1)

    def test_extract_with_markdown_fences(self):
        """Test extracting with markdown code fences"""
        json_content = json.dumps({
            "entities": [{"name": "Entity1", "semantic_type": "Type1"}]
        })
        mock_response = MagicMock()
        mock_response.message = f"```json\n{json_content}\n```"

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], "Entity1")

    def test_extract_with_invalid_json(self):
        """Test extracting with invalid JSON"""
        mock_response = MagicMock()
        mock_response.message = "This is not valid JSON"

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(result, [])

    def test_extract_with_json_in_text(self):
        """Test extracting JSON embedded in text"""
        json_content = json.dumps({
            "entities": [{"name": "Entity1", "semantic_type": "Type1"}]
        })
        mock_response = MagicMock()
        mock_response.message = f"Some text before {json_content} some text after"

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(len(result), 1)

    def test_extract_entity_without_name(self):
        """Test extracting entity without name field"""
        mock_response = MagicMock()
        mock_response.message = json.dumps({
            "entities": [
                {"semantic_type": "Type1"}  # No name field
            ]
        })

        self.mock_client.generate.return_value = mock_response

        text = "Test text"
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(result, [])

    def test_extract_multiple_entities(self):
        """Test extracting multiple entities"""
        mock_response = MagicMock()
        mock_response.message = json.dumps({
            "entities": [
                {"name": "Entity1", "semantic_type": "Type1", "mention": "Entity1"},
                {"name": "Entity2", "semantic_type": "Type2", "mention": "Entity2"}
            ]
        })

        self.mock_client.generate.return_value = mock_response

        text = "Test text with Entity1 and Entity2"
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], "Entity1")
        self.assertEqual(result[1]['name'], "Entity2")

    def test_extract_exception_handling(self):
        """Test exception handling during extraction"""
        self.mock_client.generate.side_effect = Exception("Test error")

        text = "Test text"
        result = self.extractor.extract(text, "test.json")

        self.assertEqual(result, [])

    def test_merge_duplicate_nodes_basic(self):
        """Test merging duplicate nodes with same name and semantic_type"""
        nodes = [
            {
                'name': 'Entity1',
                'semantic_type': 'Type1',
                'context_left': 'left1',
                'context_right': 'right1',
                'source_file': 'file1'
            },
            {
                'name': 'Entity1',
                'semantic_type': 'Type1',
                'context_left': 'left2',
                'context_right': 'right2',
                'source_file': 'file2'
            },
            {
                'name': 'Entity2',
                'semantic_type': 'Type2',
                'context_left': 'left3',
                'context_right': 'right3',
                'source_file': 'file3'
            }
        ]
        
        merged = self.extractor._merge_duplicate_nodes(nodes)
        
        # Should have 2 unique nodes (Entity1 merged, Entity2 separate)
        self.assertEqual(len(merged), 2)
        
        # Check Entity1 was merged
        entity1 = next(n for n in merged if n['name'] == 'Entity1')
        self.assertEqual(entity1['_duplicate_count'], 2)
        self.assertIn('left1', entity1['context_left'])
        self.assertIn('left2', entity1['context_left'])
        
        # Check Entity2 is separate
        entity2 = next(n for n in merged if n['name'] == 'Entity2')
        self.assertNotIn('_duplicate_count', entity2)

    def test_merge_duplicate_nodes_with_none(self):
        """Test merging nodes when name or semantic_type is None"""
        nodes = [
            {
                'name': None,
                'semantic_type': 'Type1',
                'context_left': 'left1',
                'context_right': 'right1'
            },
            {
                'name': None,
                'semantic_type': 'Type1',
                'context_left': 'left2',
                'context_right': 'right2'
            },
            {
                'name': 'Entity1',
                'semantic_type': None,
                'context_left': 'left3',
                'context_right': 'right3'
            }
        ]
        
        # Should not raise AttributeError
        merged = self.extractor._merge_duplicate_nodes(nodes)
        self.assertIsInstance(merged, list)

    def test_merge_duplicate_nodes_with_empty_string_context(self):
        """Test merging nodes with empty string context"""
        nodes = [
            {
                'name': 'Entity1',
                'semantic_type': 'Type1',
                'context_left': '',
                'context_right': 'right1'
            },
            {
                'name': 'Entity1',
                'semantic_type': 'Type1',
                'context_left': 'left2',
                'context_right': ''
            }
        ]
        
        merged = self.extractor._merge_duplicate_nodes(nodes)
        self.assertEqual(len(merged), 1)
        
        entity1 = merged[0]
        self.assertIn('left2', entity1['context_left'])
        self.assertIn('right1', entity1['context_right'])

