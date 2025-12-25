"""Test cases for data_loader module"""
from data_loader import extract_text_from_data, load_json_files
import json
import tempfile
from pathlib import Path
from unittest import TestCase

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataLoader(TestCase):
    """Test cases for data loading functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_json_files_empty_directory(self):
        """Test loading from empty directory"""
        result = load_json_files(self.temp_dir)
        self.assertEqual(result, [])

    def test_load_json_files_single_json(self):
        """Test loading single JSON file"""
        test_data = {"text": "Test content", "id": 1}
        json_file = self.temp_dir / "test.json"
        json_file.write_text(json.dumps(test_data), encoding='utf-8')

        result = load_json_files(self.temp_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], "Test content")
        self.assertEqual(result[0]['_source_file'], "test.json")
        self.assertIn('_file_size', result[0])

    def test_load_json_files_json_array(self):
        """Test loading JSON array"""
        test_data = [
            {"text": "Item 1", "id": 1},
            {"text": "Item 2", "id": 2}
        ]
        json_file = self.temp_dir / "test.json"
        json_file.write_text(json.dumps(test_data), encoding='utf-8')

        result = load_json_files(self.temp_dir)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], "Item 1")
        self.assertEqual(result[0]['_item_index'], 0)
        self.assertEqual(result[1]['_item_index'], 1)

    def test_load_json_files_jsonl(self):
        """Test loading JSONL file"""
        jsonl_file = self.temp_dir / "test.jsonl"
        lines = [
            '{"text": "Line 1", "id": 1}\n',
            '{"text": "Line 2", "id": 2}\n',
            '\n',  # Empty line should be skipped
            '{"text": "Line 3", "id": 3}\n'
        ]
        jsonl_file.write_text(''.join(lines), encoding='utf-8')

        result = load_json_files(self.temp_dir)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['text'], "Line 1")
        self.assertEqual(result[0]['_line_number'], 1)
        self.assertEqual(result[1]['_line_number'], 2)
        self.assertEqual(result[2]['_line_number'], 4)  # Line 3 is skipped

    def test_load_json_files_with_limit(self):
        """Test loading with limit"""
        # Create 3 JSON files
        for i in range(3):
            json_file = self.temp_dir / f"test_{i}.json"
            json_file.write_text(json.dumps({"id": i}), encoding='utf-8')

        result = load_json_files(self.temp_dir, limit=2)
        self.assertEqual(len(result), 2)

    def test_load_json_files_invalid_json(self):
        """Test handling invalid JSON"""
        json_file = self.temp_dir / "invalid.json"
        json_file.write_text("This is not valid JSON {", encoding='utf-8')

        result = load_json_files(self.temp_dir)
        # Should fallback to text
        self.assertEqual(len(result), 1)
        self.assertIn('text', result[0])
        self.assertEqual(result[0]['_format'], 'text_fallback')

    def test_extract_text_from_data_with_text_field(self):
        """Test extracting text from data with 'text' field"""
        data = {"text": "Test content", "other": "ignored"}
        result = extract_text_from_data(data)
        self.assertEqual(result, "Test content")

    def test_extract_text_from_data_with_content_field(self):
        """Test extracting text from data with 'content' field"""
        data = {"content": "Content text", "other": "ignored"}
        result = extract_text_from_data(data)
        self.assertEqual(result, "Content text")

    def test_extract_text_from_data_with_body_field(self):
        """Test extracting text from data with 'body' field"""
        data = {"body": "Body text"}
        result = extract_text_from_data(data)
        self.assertEqual(result, "Body text")

    def test_extract_text_from_data_no_text_field(self):
        """Test extracting text when no text field exists"""
        data = {
            "title": "Title",
            "author": "Author",
            "_metadata": "ignored"
        }
        result = extract_text_from_data(data)
        self.assertIn("title: Title", result)
        self.assertIn("author: Author", result)
        self.assertNotIn("_metadata", result)

    def test_extract_text_from_data_empty_dict(self):
        """Test extracting text from empty dict"""
        data = {}
        result = extract_text_from_data(data)
        self.assertEqual(result, "{}")

    def test_extract_text_from_data_only_metadata(self):
        """Test extracting text when only metadata fields exist"""
        data = {"_source_file": "test.json", "_line_number": 1}
        result = extract_text_from_data(data)
        self.assertEqual(result, "{}")
