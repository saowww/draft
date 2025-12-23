import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to sys.path
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "KGChat-03"))

# Try different import paths
try:
    from KGChat_03.backend.llm.vllm_client import VLLMClient
    from KGChat_03.backend.graph_extractor.schema import Edge, ExtractedEdges, Entity, ValidatedEntity
except ImportError:
    try:
        from backend.llm.vllm_client import VLLMClient
        from backend.graph_extractor.schema import Edge, ExtractedEdges, Entity, ValidatedEntity
    except ImportError:
        sys.path.insert(0, str(root_path / "KGChat-03"))
        from backend.llm.vllm_client import VLLMClient
        from backend.graph_extractor.schema import Edge, ExtractedEdges, Entity, ValidatedEntity

# AUTOSAR Entity Extraction Prompt
AUTOSAR_ENTITY_PROMPT = """You are an AUTOSAR expert. Extract AUTOSAR entities from the text.

Extract the following types of entities:
- Application: Application software components
- RTE: Runtime Environment components
- Basic: Basic software components
- Sensor: Sensor components
- Actuator: Actuator components
- ECU: Electronic Control Units
- Driver: Driver components
- Manager: Manager components
- Interface: Interface components
- PortInterface: Port interfaces (SenderReceiver, ClientServer, etc.)
- Signal: Signals
- Service: Services
- Protocol: Communication protocols
- Configuration: Configuration parameters

For each entity, provide:
- name: The canonical name of the entity
- semantic_type: One of the types listed above
- mention: The exact text span from the document

Return valid JSON following this schema:
{
  "entities": [
    {"name": "entity_name", "semantic_type": "type", "mention": "exact text from document"}
  ]
}

Text:
[INPUT TEXT]
"""

# AUTOSAR Edge Extraction Prompt
AUTOSAR_EDGE_PROMPT = """You are an AUTOSAR expert. Extract relationships between AUTOSAR entities from the text.

Relationship types:
- PROVIDES: Component provides an interface/service
- REQUIRES: Component requires an interface/service
- CONTAINS: Component contains sub-components
- MANAGES: Component manages another component
- CONNECTS: Components are connected
- CONFIGURES: Component configures another component
- USES: Component uses another component
- IMPLEMENTS: Component implements an interface
- DEFINES: Interface defines signals/properties

Rules:
- Only use information within the provided text
- Only create relationships between entities that appear in the provided entity list
- Return valid JSON following this schema

Text:
[INPUT TEXT]

Entities:
[ENTITIES LIST]

Return JSON with "edges" array containing relationship objects:
{
  "edges": [
    {"source": "entity1", "target": "entity2", "relation": "PROVIDES", "evidence": "supporting text"}
  ]
}
"""


class AUTOSARNodeExtractor:
    """Extract AUTOSAR entities (nodes) from text using VLLM."""

    def __init__(self, client: VLLMClient):
        self.client = client
        # Không dùng schema vì vLLM không hỗ trợ

    def extract_context(self, text: str, mention: str, window_size: int = 50) -> Tuple[str, str]:
        """Extract left and right context around a mention."""
        if not mention or mention not in text:
            return "", ""

        start_idx = text.find(mention)
        end_idx = start_idx + len(mention)

        left_start = max(0, start_idx - window_size)
        context_left = text[left_start:start_idx]

        right_end = min(len(text), end_idx + window_size)
        context_right = text[end_idx:right_end]

        return context_left, context_right

    def extract(self, text: str, file_name: str = "unknown") -> List[Dict]:
        """Extract AUTOSAR entities from text."""
        if not text or not text.strip():
            return []

        prompt = AUTOSAR_ENTITY_PROMPT.replace("[INPUT TEXT]", text)

        try:
            # vLLM không hỗ trợ schema, chỉ dùng prompt và parse JSON
            resp = self.client.generate(prompt=prompt, format=None)

            # Parse JSON từ response string
            if isinstance(resp, str):
                import re
                # Remove markdown code fences nếu có
                resp_clean = re.sub(r'^```json\s*', '',
                                    resp, flags=re.MULTILINE)
                resp_clean = re.sub(r'```$', '', resp_clean,
                                    flags=re.MULTILINE).strip()
                # Tìm JSON object
                try:
                    resp = json.loads(resp_clean)
                except json.JSONDecodeError:
                    # Thử tìm { đầu tiên và } cuối cùng
                    start = resp_clean.find('{')
                    end = resp_clean.rfind('}') + 1
                    if start >= 0 and end > start:
                        resp = json.loads(resp_clean[start:end])
                    else:
                        raise ValueError("Could not parse JSON from response")

            entities = resp.get("entities", [])

            # Process entities and add context
            output = []
            for entity in entities:
                name = entity.get("name", "").strip()
                semantic_type = entity.get("semantic_type", "").strip()
                mention = entity.get("mention", name).strip()

                if name:
                    context_left, context_right = self.extract_context(
                        text, mention)
                    output.append({
                        "name": name,
                        "semantic_type": semantic_type,
                        "mention": mention,
                        "context_left": context_left,
                        "context_right": context_right,
                        "source_file": file_name,
                        "level": "Level 1"
                    })

            return output
        except Exception as e:
            print(f"Error extracting entities: {e}")
            import traceback
            traceback.print_exc()
            return []


class AUTOSAREdgeExtractor:
    """Extract AUTOSAR relationships (edges) from text using VLLM."""

    def __init__(self, client: VLLMClient):
        self.client = client
        # Không dùng schema vì vLLM không hỗ trợ

    def extract(self, text: str, nodes: List[Dict], file_name: str = "unknown") -> ExtractedEdges:
        """Extract AUTOSAR relationships from text using entities."""
        if not text or not text.strip() or not nodes:
            return ExtractedEdges(edges=[])

        # Prepare entity list for prompt
        prompt_entities = []
        for node in nodes:
            prompt_entities.append({
                "name": node.get("name", ""),
                "semantic_type": node.get("semantic_type", "")
            })

        prompt = (
            AUTOSAR_EDGE_PROMPT
            .replace("[INPUT TEXT]", text)
            .replace("[ENTITIES LIST]", json.dumps({"entities": prompt_entities}, ensure_ascii=False))
        )

        try:
            # vLLM không hỗ trợ schema, chỉ dùng prompt và parse JSON
            resp = self.client.generate(prompt=prompt, format=None)

            # Parse JSON từ response string
            if isinstance(resp, str):
                import re
                # Remove markdown code fences nếu có
                resp_clean = re.sub(r'^```json\s*', '',
                                    resp, flags=re.MULTILINE)
                resp_clean = re.sub(r'```$', '', resp_clean,
                                    flags=re.MULTILINE).strip()
                # Tìm JSON object
                try:
                    resp = json.loads(resp_clean)
                except json.JSONDecodeError:
                    # Thử tìm { đầu tiên và } cuối cùng
                    start = resp_clean.find('{')
                    end = resp_clean.rfind('}') + 1
                    if start >= 0 and end > start:
                        resp = json.loads(resp_clean[start:end])
                    else:
                        raise ValueError("Could not parse JSON from response")

            # Convert dict to ExtractedEdges
            if isinstance(resp, dict):
                edges_result = ExtractedEdges(**resp)
            elif isinstance(resp, ExtractedEdges):
                edges_result = resp
            else:
                edges_result = ExtractedEdges(edges=[])

            # Add source_file to edges
            for edge in edges_result.edges:
                edge_dict = edge.dict()
                edge_dict['source_file'] = file_name

            return edges_result
        except Exception as e:
            print(f"Error extracting edges: {e}")
            import traceback
            traceback.print_exc()
            return ExtractedEdges(edges=[])


def load_json_files(data_dir: Path, limit: int = 5) -> List[Dict]:
    """Load JSON files from directory. If file is JSONL, read lines."""
    json_files = sorted(data_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return []

    all_data = []

    # Process up to limit files
    for json_file in json_files[:limit]:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                # Try to read as JSON first
                try:
                    content = f.read()
                    # Check if it's JSONL (each line is a JSON object)
                    if content.strip().startswith('{') and '\n' in content:
                        # JSONL format
                        f.seek(0)
                        for line_num, line in enumerate(f):
                            if line.strip():
                                try:
                                    data = json.loads(line.strip())
                                    data['_source_file'] = json_file.name
                                    data['_line_number'] = line_num
                                    all_data.append(data)
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Regular JSON
                        data = json.loads(content)
                        if isinstance(data, list):
                            for item in data:
                                item['_source_file'] = json_file.name
                                all_data.append(item)
                        else:
                            data['_source_file'] = json_file.name
                            all_data.append(data)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as text
                    f.seek(0)
                    text = f.read()
                    all_data.append({
                        'text': text,
                        '_source_file': json_file.name
                    })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

    return all_data


def extract_text_from_data(data: Dict) -> str:
    """Extract text field from JSON data."""
    # Try common text fields
    for field in ['text', 'content', 'body', 'description', 'document']:
        if field in data and isinstance(data[field], str):
            return data[field]

    # If no text field, convert entire dict to string (excluding metadata)
    text_parts = []
    for key, value in data.items():
        if not key.startswith('_') and isinstance(value, str):
            text_parts.append(f"{key}: {value}")

    return " ".join(text_parts) if text_parts else str(data)


def save_nodes_csv(nodes: List[Dict], filepath: Path, append: bool = False):
    """Save nodes to CSV file."""
    if not nodes:
        return

    fieldnames = set()
    for node in nodes:
        fieldnames.update(node.keys())

    ordered_fields = ['name', 'semantic_type', 'source_file', 'level']
    other_fields = [f for f in fieldnames if f not in ordered_fields]
    fieldnames = ordered_fields + other_fields

    mode = "a" if append else "w"
    write_header = not append or not filepath.exists() or filepath.stat().st_size == 0

    with filepath.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(nodes)


def save_edges_csv(edges: List[Dict], filepath: Path, append: bool = False):
    """Save edges to CSV file."""
    if not edges:
        return

    fieldnames = set()
    for edge in edges:
        fieldnames.update(edge.keys())

    ordered_fields = ['source', 'relation', 'target', 'source_file']
    other_fields = [f for f in fieldnames if f not in ordered_fields]
    fieldnames = ordered_fields + other_fields

    mode = "a" if append else "w"
    write_header = not append or not filepath.exists() or filepath.stat().st_size == 0

    with filepath.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(edges)


def main():
    # Configuration
    config = {
        "base_url": "http://localhost:8887/v1",
        "model": "GLM-4.6-AWQ",
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 2000
    }

    client = VLLMClient(config)

    # Initialize extractors
    node_extractor = AUTOSARNodeExtractor(client)
    edge_extractor = AUTOSAREdgeExtractor(client)

    # Load data
    data_dir = Path("autosardat")
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    print(f"Loading JSON files from {data_dir}...")
    json_data = load_json_files(data_dir, limit=5)

    if not json_data:
        print("No data loaded")
        return

    print(f"Loaded {len(json_data)} records from JSON files")

    # Output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    nodes_path = output_dir / "autosar_nodes.csv"
    edges_path = output_dir / "autosar_edges.csv"

    # Clear existing files
    if nodes_path.exists():
        nodes_path.unlink()
    if edges_path.exists():
        edges_path.unlink()

        all_nodes = []
        all_edges = []

    # Process each record
    for idx, data in enumerate(json_data):
        print(f"\nProcessing record {idx + 1}/{len(json_data)}...")

        text = extract_text_from_data(data)
        source_file = data.get('_source_file', 'unknown')

        if not text or len(text.strip()) < 10:
            print(f"Skipping record {idx + 1}: text too short")
            continue

        print(f"Text length: {len(text)} characters")

        # Extract nodes
        print("Extracting nodes...")
        start_time = time.time()
        nodes = node_extractor.extract(text, file_name=source_file)
        node_time = time.time() - start_time
        print(f"Extracted {len(nodes)} nodes in {node_time:.2f}s")

        # Extract edges
        edges_result = None
        if nodes:
            print("Extracting edges...")
            start_time = time.time()
            edges_result = edge_extractor.extract(
                text, nodes, file_name=source_file)
            edge_time = time.time() - start_time
            print(
                f"Extracted {len(edges_result.edges)} edges in {edge_time:.2f}s")

            # Convert edges to dict
            edges_dict = [edge.dict() for edge in edges_result.edges]
            all_edges.extend(edges_dict)
        else:
            print("No nodes found, skipping edge extraction")

        all_nodes.extend(nodes)

        # Save incrementally
        save_nodes_csv(nodes, nodes_path, append=True)
        if edges_result and edges_result.edges:
            save_edges_csv(edges_dict, edges_path, append=True)

    print("\n=== Summary ===")
    print(f"Total nodes extracted: {len(all_nodes)}")
    print(f"Total edges extracted: {len(all_edges)}")
    print(f"Nodes saved to: {nodes_path}")
    print(f"Edges saved to: {edges_path}")


if __name__ == "__main__":
    main()
