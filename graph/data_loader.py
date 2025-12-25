"""Data loading utilities for AUTOSAR extractor"""
import json
from pathlib import Path
from typing import Dict, List


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
