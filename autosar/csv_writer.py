"""CSV writing utilities for AUTOSAR extractor"""
import csv
from pathlib import Path
from typing import Dict, List


def save_nodes_csv(nodes: List[Dict], filepath: Path, append: bool = False):
    """Save nodes to CSV file"""
    if not nodes:
        return
    fieldnames = set()
    for node in nodes:
        fieldnames.update(node.keys())
    ordered_fields = ['name', 'semantic_type', 'source_file', 'level']
    other_fields = [f for f in fieldnames if f not in ordered_fields]
    fieldnames = ordered_fields + other_fields  # fix

    mode = "a" if append else "w"
    write_header = not append or not filepath.exists() or filepath.stat().st_size == 0
    with filepath.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(nodes)


def save_edges_csv(edges: List[Dict], filepath: Path, append: bool = False):
    """Save edges to CSV file"""
    if not edges:
        return
    fieldnames = set()
    for edge in edges:
        fieldnames.update(edge.keys())
    ordered_fields = ['source', 'relation', 'target', 'source_file']
    other_fields = [f for f in fieldnames if f not in ordered_fields]
    fieldnames = ordered_fields + other_fields  # fix
    mode = "a" if append else "w"
    write_header = not append or not filepath.exists() or filepath.stat().st_size == 0
    with filepath.open(mode, newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:  # fix
            writer.writeheader()
        writer.writerows(edges)
