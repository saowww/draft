"""Test cases for csv_writer module"""
from csv_writer import save_edges_csv, save_nodes_csv
import csv
import tempfile
from pathlib import Path
from unittest import TestCase

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCSVWriter(TestCase):
    """Test cases for CSV writing functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_nodes_csv_empty_list(self):
        """Test saving empty nodes list"""
        filepath = self.temp_dir / "nodes.csv"
        save_nodes_csv([], filepath)
        self.assertFalse(filepath.exists())

    def test_save_nodes_csv_single_node(self):
        """Test saving single node"""
        filepath = self.temp_dir / "nodes.csv"
        nodes = [{
            "name": "TestNode",
            "semantic_type": "Entity",
            "source_file": "test.json"
        }]
        save_nodes_csv(nodes, filepath)

        self.assertTrue(filepath.exists())
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['name'], "TestNode")
            self.assertEqual(rows[0]['semantic_type'], "Entity")

    def test_save_nodes_csv_multiple_nodes(self):
        """Test saving multiple nodes"""
        filepath = self.temp_dir / "nodes.csv"
        nodes = [
            {"name": "Node1", "semantic_type": "Type1", "source_file": "file1.json"},
            {"name": "Node2", "semantic_type": "Type2", "source_file": "file2.json"}
        ]
        save_nodes_csv(nodes, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)

    def test_save_nodes_csv_ordered_fields(self):
        """Test that ordered fields appear first"""
        filepath = self.temp_dir / "nodes.csv"
        nodes = [{
            "name": "Test",
            "semantic_type": "Type",
            "source_file": "test.json",
            "extra_field": "extra"
        }]
        save_nodes_csv(nodes, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            # Check that ordered fields come first
            self.assertEqual(fieldnames[0], 'name')
            self.assertEqual(fieldnames[1], 'semantic_type')
            self.assertEqual(fieldnames[2], 'source_file')
            self.assertEqual(fieldnames[3], 'level')

    def test_save_nodes_csv_append_mode(self):
        """Test appending to existing file"""
        filepath = self.temp_dir / "nodes.csv"
        nodes1 = [{"name": "Node1", "semantic_type": "Type1"}]
        nodes2 = [{"name": "Node2", "semantic_type": "Type2"}]

        save_nodes_csv(nodes1, filepath)
        save_nodes_csv(nodes2, filepath, append=True)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)

    def test_save_edges_csv_empty_list(self):
        """Test saving empty edges list"""
        filepath = self.temp_dir / "edges.csv"
        save_edges_csv([], filepath)
        self.assertFalse(filepath.exists())

    def test_save_edges_csv_single_edge(self):
        """Test saving single edge"""
        filepath = self.temp_dir / "edges.csv"
        edges = [{
            "source": "Node1",
            "relation": "relates_to",
            "target": "Node2",
            "source_file": "test.json"
        }]
        save_edges_csv(edges, filepath)

        self.assertTrue(filepath.exists())
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['source'], "Node1")
            self.assertEqual(rows[0]['relation'], "relates_to")
            self.assertEqual(rows[0]['target'], "Node2")

    def test_save_edges_csv_multiple_edges(self):
        """Test saving multiple edges"""
        filepath = self.temp_dir / "edges.csv"
        edges = [
            {"source": "Node1", "relation": "rel1", "target": "Node2"},
            {"source": "Node2", "relation": "rel2", "target": "Node3"}
        ]
        save_edges_csv(edges, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)

    def test_save_edges_csv_ordered_fields(self):
        """Test that ordered fields appear first"""
        filepath = self.temp_dir / "edges.csv"
        edges = [{
            "source": "Node1",
            "relation": "rel",
            "target": "Node2",
            "source_file": "test.json",
            "extra_field": "extra"
        }]
        save_edges_csv(edges, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            # Check that ordered fields come first
            self.assertEqual(fieldnames[0], 'source')
            self.assertEqual(fieldnames[1], 'relation')
            self.assertEqual(fieldnames[2], 'target')
            self.assertEqual(fieldnames[3], 'source_file')

    def test_save_edges_csv_append_mode(self):
        """Test appending to existing file"""
        filepath = self.temp_dir / "edges.csv"
        edges1 = [{"source": "N1", "relation": "r1", "target": "N2"}]
        edges2 = [{"source": "N3", "relation": "r2", "target": "N4"}]

        save_edges_csv(edges1, filepath)
        save_edges_csv(edges2, filepath, append=True)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
