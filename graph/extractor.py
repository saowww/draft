from backend.graph_extractor.prompts import AUTOSAR_ENTITY_PROMPT, AUTOSAR_EDGE_PROMPT
from vllm_client import VLLMClient
from backend.graph_extractor.schema import Edge, ExtractedEdges, Entity, ValidatedEntity
import sys
import time
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add sys.path nếu cần
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))


print("Success lib")


def load_json_files(data_dir: Path, limit: int = None) -> List[Dict]:
    """
    Production-ready JSON/JSONL loader
    - JSONL: line-by-line parsing
    - JSON: single object/array  
    - Mixed formats supported
    - Robust error handling
    """
    json_files = sorted(data_dir.glob("*.json")) + \
        sorted(data_dir.glob("*.jsonl"))

    if not json_files:
        print(f"No JSON/JSONL files found in {data_dir}")
        return []

    print(f"Found {len(json_files)} files")
    all_data = []
    processed_count = 0

    for json_file in json_files:
        if limit and processed_count >= limit:
            print(f"Reached limit {limit}")
            break

        print(f"Processing {json_file.name}...")
        file_count = 0

        try:
            if json_file.suffix.lower() == '.jsonl':
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            data['_source_file'] = json_file.name
                            data['_line_number'] = line_num
                            data['_file_size'] = json_file.stat().st_size
                            all_data.append(data)
                            file_count += 1
                        except json.JSONDecodeError as e:
                            print(f"Line {line_num}: {e}")
                            continue

            else:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            item['_source_file'] = json_file.name
                            item['_item_index'] = i
                            item['_file_size'] = json_file.stat().st_size
                            all_data.append(item)
                            file_count += 1
                    else:
                        data['_source_file'] = json_file.name
                        data['_file_size'] = json_file.stat().st_size
                        all_data.append(data)
                        file_count += 1

        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_data.append({
                        'text': text[:2000],
                        '_source_file': json_file.name,
                        '_format': 'text_fallback'
                    })
                    file_count += 1
            except Exception as fallback_e:
                print(f"Fallback failed: {fallback_e}")

        except Exception as e:
            print(f"Error: {e}")
            continue

        processed_count += 1
        print(f"Loaded {file_count} records from {json_file.name}")

    print(f"Total: {len(all_data)} records from {processed_count} files")
    return all_data


class AUTOSARNodeExtractor:
    """Extract Autosar entities (nodes) from text using VLLM"""

    def __init__(self, client: VLLMClient):
        self.client = client

    def extract_context(self, text: str, mention: str, window_size: int = 50) -> Tuple[str, str]:
        """Extract left and right context around a mention"""
        if not mention or mention not in text:
            return "", ""
        start_idx = text.find(mention)  # ✅ FIXED: text.find()
        end_idx = start_idx + len(mention)
        left_start = max(0, start_idx - window_size)
        context_left = text[left_start:start_idx]
        right_end = min(len(text), end_idx + window_size)
        context_right = text[end_idx:right_end]
        return context_left, context_right

    def extract(self, text: str, file_name: str = 'unknown') -> List[Dict]:
        """Extract AUTOSAR entities from text."""
        if not text or not text.strip():
            return []

        prompt = AUTOSAR_ENTITY_PROMPT.replace("[INPUT TEXT]", text)

        try:
            resp = self.client.generate(
                user_prompt=prompt,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000
            )

            # Extract text content from different response formats
            if hasattr(resp, 'message'):  # fix
                # LLMResponse object (from VLLMClient)
                resp_text = resp.message  # fix
            elif hasattr(resp, 'choices') and resp.choices:
                # vLLM OpenAI-compatible format
                resp_text = resp.choices[0].message.content
            elif hasattr(resp, 'text'):
                # Direct text response
                resp_text = resp.text
            elif isinstance(resp, str):
                resp_text = resp
            else:
                print(f"Unknown response format: {type(resp)}")
                return []

            # Clean markdown code fences
            resp_clean = re.sub(r'^```[a-z]*\n?', '',  # fix
                                resp_text, flags=re.MULTILINE)
            resp_clean = re.sub(r'```$', '', resp_clean,
                                flags=re.MULTILINE).strip()

            # Parse JSON
            try:
                resp_dict = json.loads(resp_clean)
            except json.JSONDecodeError:
                # Fallback: extract JSON object from string
                start = resp_clean.find('{')
                end = resp_clean.rfind('}') + 1
                if start >= 0 and end > start:
                    resp_dict = json.loads(resp_clean[start:end])
                else:
                    print("Could not parse JSON from response")
                    return []

            entities = resp_dict.get("entities", [])

            # Add context to entities
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

                    })

            return output

        except Exception as e:
            print(f"Error extracting entities: {e}")
            import traceback
            traceback.print_exc()
            return []


class AUTOSAREdgeExtractor:
    def __init__(self, client: VLLMClient):
        self.client = client

    def extract(self, text: str, nodes: List[Dict], file_name: str = "unknow") -> ExtractedEdges:
        """Extract AUTOSAR relationships from text using entities"""
        if not text or not text.strip() or not nodes:
            return ExtractedEdges(edges=[])

        prompt_entities = []
        for node in nodes:
            prompt_entities.append({
                "name": node.get("name", ""),
                "semantic_type": node.get("semantic_type", "")

            })
        prompt = (
            AUTOSAR_EDGE_PROMPT
            .replace("[INPUT TEXT]", text)
            .replace("ENTITIES LIST", json.dumps({"entities": prompt_entities}, ensure_ascii=False))
        )
        try:
            resp = self.client.generate(
                user_prompt=prompt,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000
            )

            # Extract text content from different response formats
            if hasattr(resp, 'message'):
                # LLMResponse object (from VLLMClient)
                resp_text = resp.message
            elif hasattr(resp, 'choices') and resp.choices:
                # vLLM OpenAI-compatible format
                resp_text = resp.choices[0].message.content
            elif hasattr(resp, 'text'):
                # Direct text response
                resp_text = resp.text
            elif isinstance(resp, str):
                resp_text = resp
            else:
                print(f"Unknown response format: {type(resp)}")
                return ExtractedEdges(edges=[])

            # Clean markdown code fences
            resp_clean = re.sub(r'^```[a-z]*\n?', '',
                                resp_text, flags=re.MULTILINE)
            resp_clean = re.sub(r'```$', '', resp_clean,
                                flags=re.MULTILINE).strip()

            # Parse JSON từ response string
            try:
                resp_dict = json.loads(resp_clean)  # fix
            except json.JSONDecodeError:
                start = resp_clean.find('{')
                end = resp_clean.rfind('}') + 1
                if start >= 0 and end > start:
                    resp_dict = json.loads(resp_clean[start:end])
                else:
                    raise ValueError("Could not parse JSON from response")

            # Convert dict to ExtractedEdges
            if isinstance(resp_dict, dict):  # fix
                edges_result = ExtractedEdges(**resp_dict)
            elif isinstance(resp_dict, ExtractedEdges):
                edges_result = resp_dict
            else:
                edges_result = ExtractedEdges(edges=[])
            # Add source_file to edges
            for edge in edges_result.edges:
                edge_dict = edge.dict()
                edge_dict['source_file'] = file_name
            return edges_result
        except Exception as e:
            print(f"Error etracting edges: {e}")
            import traceback
            traceback.print_exc()
            return ExtractedEdges(edges=[])


def extract_text_from_data(data: Dict) -> str:
    """Extract text field from JSON data"""
    for field in ['text', 'content', 'body', 'description', 'document']:
        if field in data and isinstance(data[field], str):
            return data[field]

    # if no textfield convert entire dict to string (excluding metadata)
    text_parts = []
    for key, value in data.items():
        if not key.startswith("_") and isinstance(value, str):
            text_parts.append(f"{key}: {value}")
    return " ".join(text_parts) if text_parts else str(data)  # fix


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


def main():
    client = VLLMClient()
    data_dir = Path("/mnt/hps/dungmt19_workspace/KBS/data")
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    print(f"Loading JSON files from {data_dir}...")
    json_data = load_json_files(data_dir, limit=5)

    print(f"Loaded {len(json_data)} records")
    if json_data:
        print("Sample data:")
        print(json.dumps(json_data[0], indent=2,
              ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    # con vllmclient
    client = VLLMClient()
    print("Health check:", client.health_check())
    node_extractor = AUTOSARNodeExtractor(client)
    edge_extractor = AUTOSAREdgeExtractor(client)
    data_dir = Path("/mnt/hps/dungmt19_workspace/KBS/data")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")

    print(f"Loading Json/Jsonl files from {data_dir} ... ")
    start_time = time.time()
    json_data = load_json_files(data_dir, limit=5)
    load_time = time.time() - start_time
    print(f"Loaded {len(json_data)} records in {load_time:.2f}s")
    if not json_data:
        print("No data loaded")

    # Output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    nodes_path = output_dir/"autosar_nodes.csv"
    edges_path = output_dir/"autosar_edges.csv"
    # Clear existing files
    if nodes_path.exists():
        nodes_path.unlink()
    if edges_path.exists():
        edges_path.unlink()
    all_nodes = []
    all_edges = []
    for idx, data in enumerate(json_data):
        print(f"Processing record {idx+1}/{len(json_data)}...")
        print(f"Source: {data.get('__source_file', 'unknown')}")
        text = extract_text_from_data(data)
        source_file = data.get('_source_file', 'unknown')

        if not text or len(text.strip()) < 10:
            print(f"Skipping: text too short ({len(text)} chars)")
            continue
        print(f"Text length: {len(text)} characters")

        # Extract nodes
        print("Extracting nodes...")
        node_start = time.time()
        nodes = node_extractor.extract(text, file_name=source_file)
        node_time = time.time() - node_start
        print(f"{len(nodes)} nodes in {node_time:.2f}s")

        # Extract edges
        if nodes:
            print("Extracting edges...")
            edge_start = time.time()
            edges_result = edge_extractor.extract(
                text, nodes, file_name=source_file)
            edge_time = time.time() - edge_start
            print(f"{len(edges_result.edges)} edges in {edge_time:.2f}s")
            # Convert edges to dict
            edges_dict = [edge.dict() for edge in edges_result.edges]
            all_edges.extend(edges_dict)
        else:
            print("No nodes, skipping edges")
        all_nodes.extend(nodes)
        save_nodes_csv(nodes, nodes_path, append=True)
        if 'edges_dict' in locals() and edges_dict:
            save_edges_csv(edges_dict, edges_path, append=True)
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE!")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Total edges: {len(all_edges)}")
    print(f"Nodes saved: {nodes_path}")
    print(f"Edges saved: {edges_path}")
    print("\n" + "=" * 50)
if __name__ == "__main__":
    main()


def load_json_files(data_dir: Path, limit: int = None) -> List[Dict]:
    """
    Production-ready JSON/JSONL loader
    - JSONL: line-by-line parsing
    - JSON: single object/array  
    - Mixed formats supported
    - Robust error handling
    """
    json_files = sorted(data_dir.glob("*.json")) + \
        sorted(data_dir.glob("*.jsonl"))

    if not json_files:
        print(f"No JSON/JSONL files found in {data_dir}")
        return []

    print(f"Found {len(json_files)} files")
    all_data = []
    processed_count = 0

    for json_file in json_files:
        if limit and processed_count >= limit:
            print(f"Reached limit {limit}")
            break

        print(f"Processing {json_file.name}...")
        file_count = 0

        try:
            if json_file.suffix.lower() == '.jsonl':
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            data['_source_file'] = json_file.name
                            data['_line_number'] = line_num
                            data['_file_size'] = json_file.stat().st_size
                            all_data.append(data)
                            file_count += 1
                        except json.JSONDecodeError as e:
                            print(f"Line {line_num}: {e}")
                            continue

            else:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            item['_source_file'] = json_file.name
                            item['_item_index'] = i
                            item['_file_size'] = json_file.stat().st_size
                            all_data.append(item)
                            file_count += 1
                    else:
                        data['_source_file'] = json_file.name
                        data['_file_size'] = json_file.stat().st_size
                        all_data.append(data)
                        file_count += 1

        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_data.append({
                        'text': text[:2000],
                        '_source_file': json_file.name,
                        '_format': 'text_fallback'
                    })
                    file_count += 1
            except Exception as fallback_e:
                print(f"Fallback failed: {fallback_e}")

        except Exception as e:
            print(f"Error: {e}")
            continue

        processed_count += 1
        print(f"Loaded {file_count} records from {json_file.name}")

    print(f"Total: {len(all_data)} records from {processed_count} files")
    return all_data


class AUTOSARNodeExtractor:
    """Extract Autosar entities (nodes) from text using VLLM"""

    def __init__(self, client: VLLMClient):
        self.client = client

    def extract_context(self, text: str, mention: str, window_size: int = 50) -> Tuple[str, str]:
        """Extract left and right context around a mention"""
        if not mention or mention not in text:
            return "", ""
        start_idx = text.find(mention)  # ✅ FIXED: text.find()
        end_idx = start_idx + len(mention)
        left_start = max(0, start_idx - window_size)
        context_left = text[left_start:start_idx]
        right_end = min(len(text), end_idx + window_size)
        context_right = text[end_idx:right_end]
        return context_left, context_right

    def extract(self, text: str, file_name: str = 'unknown') -> List[Dict]:
        """Extract AUTOSAR entities from text."""
        if not text or not text.strip():
            return []

        prompt = AUTOSAR_ENTITY_PROMPT.replace("[INPUT TEXT]", text)

        try:
            resp = self.client.generate(
                user_prompt=prompt,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000
            )

            # Extract text content from different response formats
            if hasattr(resp, 'message'):  # fix
                # LLMResponse object (from VLLMClient)
                resp_text = resp.message  # fix
            elif hasattr(resp, 'choices') and resp.choices:
                # vLLM OpenAI-compatible format
                resp_text = resp.choices[0].message.content
            elif hasattr(resp, 'text'):
                # Direct text response
                resp_text = resp.text
            elif isinstance(resp, str):
                resp_text = resp
            else:
                print(f"Unknown response format: {type(resp)}")
                return []

            # Clean markdown code fences
            resp_clean = re.sub(r'^```[a-z]*\n?', '',  # fix
                                resp_text, flags=re.MULTILINE)
            resp_clean = re.sub(r'```$', '', resp_clean,
                                flags=re.MULTILINE).strip()

            # Parse JSON
            try:
                resp_dict = json.loads(resp_clean)
            except json.JSONDecodeError:
                # Fallback: extract JSON object from string
                start = resp_clean.find('{')
                end = resp_clean.rfind('}') + 1
                if start >= 0 and end > start:
                    resp_dict = json.loads(resp_clean[start:end])
                else:
                    print("Could not parse JSON from response")
                    return []

            entities = resp_dict.get("entities", [])

            # Add context to entities
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

                    })

            return output

        except Exception as e:
            print(f"Error extracting entities: {e}")
            import traceback
            traceback.print_exc()
            return []


class AUTOSAREdgeExtractor:
    def __init__(self, client: VLLMClient):
        self.client = client

    def extract(self, text: str, nodes: List[Dict], file_name: str = "unknow") -> ExtractedEdges:
        """Extract AUTOSAR relationships from text using entities"""
        if not text or not text.strip() or not nodes:
            return ExtractedEdges(edges=[])

        prompt_entities = []
        for node in nodes:
            prompt_entities.append({
                "name": node.get("name", ""),
                "semantic_type": node.get("semantic_type", "")

            })
        prompt = (
            AUTOSAR_EDGE_PROMPT
            .replace("[INPUT TEXT]", text)
            .replace("ENTITIES LIST", json.dumps({"entities": prompt_entities}, ensure_ascii=False))
        )
        try:
            resp = self.client.generate(
                user_prompt=prompt,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000
            )

            # Extract text content from different response formats
            if hasattr(resp, 'message'):
                # LLMResponse object (from VLLMClient)
                resp_text = resp.message
            elif hasattr(resp, 'choices') and resp.choices:
                # vLLM OpenAI-compatible format
                resp_text = resp.choices[0].message.content
            elif hasattr(resp, 'text'):
                # Direct text response
                resp_text = resp.text
            elif isinstance(resp, str):
                resp_text = resp
            else:
                print(f"Unknown response format: {type(resp)}")
                return ExtractedEdges(edges=[])

            # Clean markdown code fences
            resp_clean = re.sub(r'^```[a-z]*\n?', '',
                                resp_text, flags=re.MULTILINE)
            resp_clean = re.sub(r'```$', '', resp_clean,
                                flags=re.MULTILINE).strip()

            # Parse JSON từ response string
            try:
                resp_dict = json.loads(resp_clean)  # fix
            except json.JSONDecodeError:
                start = resp_clean.find('{')
                end = resp_clean.rfind('}') + 1
                if start >= 0 and end > start:
                    resp_dict = json.loads(resp_clean[start:end])
                else:
                    raise ValueError("Could not parse JSON from response")

            # Convert dict to ExtractedEdges
            if isinstance(resp_dict, dict):  # fix
                edges_result = ExtractedEdges(**resp_dict)
            elif isinstance(resp_dict, ExtractedEdges):
                edges_result = resp_dict
            else:
                edges_result = ExtractedEdges(edges=[])
            # Add source_file to edges
            for edge in edges_result.edges:
                edge_dict = edge.dict()
                edge_dict['source_file'] = file_name
            return edges_result
        except Exception as e:
            print(f"Error etracting edges: {e}")
            import traceback
            traceback.print_exc()
            return ExtractedEdges(edges=[])


def extract_text_from_data(data: Dict) -> str:
    """Extract text field from JSON data"""
    for field in ['text', 'content', 'body', 'description', 'document']:
        if field in data and isinstance(data[field], str):
            return data[field]

    # if no textfield convert entire dict to string (excluding metadata)
    text_parts = []
    for key, value in data.items():
        if not key.startswith("_") and isinstance(value, str):
            text_parts.append(f"{key}: {value}")
    return " ".join(text_parts) if text_parts else str(data)  # fix


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


def main():
    client = VLLMClient()
    data_dir = Path("/mnt/hps/dungmt19_workspace/KBS/data")
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    print(f"Loading JSON files from {data_dir}...")
    json_data = load_json_files(data_dir, limit=5)

    print(f"Loaded {len(json_data)} records")
    if json_data:
        print("Sample data:")
        print(json.dumps(json_data[0], indent=2,
              ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    # con vllmclient
    client = VLLMClient()
    print("Health check:", client.health_check())
    node_extractor = AUTOSARNodeExtractor(client)
    edge_extractor = AUTOSAREdgeExtractor(client)
    data_dir = Path("/mnt/hps/dungmt19_workspace/KBS/data")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")

    print(f"Loading Json/Jsonl files from {data_dir} ... ")
    start_time = time.time()
    json_data = load_json_files(data_dir, limit=5)
    load_time = time.time() - start_time
    print(f"Loaded {len(json_data)} records in {load_time:.2f}s")
    if not json_data:
        print("No data loaded")

    # Output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    nodes_path = output_dir/"autosar_nodes.csv"
    edges_path = output_dir/"autosar_edges.csv"
    # Clear existing files
    if nodes_path.exists():
        nodes_path.unlink()
    if edges_path.exists():
        edges_path.unlink()
    all_nodes = []
    all_edges = []
    for idx, data in enumerate(json_data):
        print(f"Processing record {idx+1}/{len(json_data)}...")
        print(f"Source: {data.get('__source_file', 'unknown')}")
        text = extract_text_from_data(data)
        source_file = data.get('_source_file', 'unknown')

        if not text or len(text.strip()) < 10:
            print(f"Skipping: text too short ({len(text)} chars)")
            continue
        print(f"Text length: {len(text)} characters")

        # Extract nodes
        print("Extracting nodes...")
        node_start = time.time()
        nodes = node_extractor.extract(text, file_name=source_file)
        node_time = time.time() - node_start
        print(f"{len(nodes)} nodes in {node_time:.2f}s")

        # Extract edges
        if nodes:
            print("Extracting edges...")
            edge_start = time.time()
            edges_result = edge_extractor.extract(
                text, nodes, file_name=source_file)
            edge_time = time.time() - edge_start
            print(f"{len(edges_result.edges)} edges in {edge_time:.2f}s")
            # Convert edges to dict
            edges_dict = [edge.dict() for edge in edges_result.edges]
            all_edges.extend(edges_dict)
        else:
            print("No nodes, skipping edges")
        all_nodes.extend(nodes)
        save_nodes_csv(nodes, nodes_path, append=True)
        if 'edges_dict' in locals() and edges_dict:
            save_edges_csv(edges_dict, edges_path, append=True)
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE!")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Total edges: {len(all_edges)}")
    print(f"Nodes saved: {nodes_path}")
    print(f"Edges saved: {edges_path}")
    print("\n" + "=" * 50)
if __name__ == "__main__":
    main()
