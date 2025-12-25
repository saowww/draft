"""Data loading utilities for AUTOSAR extractor"""
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_json_files(data_dir: Path, limit: int = None) -> List[Dict]:
    """
    Đơn giản hóa: Load JSON/JSONL files và giữ lại TẤT CẢ các fields từ JSON

    Args:
        data_dir: Thư mục chứa file JSON/JSONL
        limit: Giới hạn số lượng file (None = không giới hạn)

    Returns:
        List các dict, mỗi dict chứa TẤT CẢ fields từ JSON + _source_file
    """
    json_files = sorted(data_dir.glob("*.json")) + \
        sorted(data_dir.glob("*.jsonl"))

    if not json_files:
        print(f"Không tìm thấy file JSON/JSONL trong {data_dir}")
        return []

    print(f"Tìm thấy {len(json_files)} files")
    all_data = []
    processed_count = 0

    for json_file in json_files:
        if limit and processed_count >= limit:
            break

        print(f"Đang xử lý {json_file.name}...")
        file_count = 0

        try:
            # Xử lý JSONL (mỗi dòng là một JSON object)
            if json_file.suffix.lower() == '.jsonl':
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            # Thêm thông tin file, giữ lại TẤT CẢ fields từ JSON
                            data['_source_file'] = json_file.name
                            all_data.append(data)
                            file_count += 1
                        except json.JSONDecodeError as e:
                            print(f"  Dòng {line_num} lỗi: {e}")
                            continue

            # Xử lý JSON (có thể là object hoặc array)
            else:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        # Nếu là array, xử lý từng item
                        for item in data:
                            item['_source_file'] = json_file.name
                            all_data.append(item)
                            file_count += 1
                    else:
                        # Nếu là object đơn
                        data['_source_file'] = json_file.name
                        all_data.append(data)
                        file_count += 1

        except Exception as e:
            print(f"  Lỗi khi đọc file {json_file.name}: {e}")
            continue

        processed_count += 1
        print(f"  Đã load {file_count} records từ {json_file.name}")

    print(f"Tổng cộng: {len(all_data)} records từ {processed_count} files")
    return all_data


def extract_text_from_data(data: Dict) -> str:
    """Lấy text từ JSON data"""
    # Ưu tiên field 'text'
    if 'text' in data and isinstance(data['text'], str):
        return data['text']

    # Thử các field khác
    for field in ['content', 'body', 'description', 'document']:
        if field in data and isinstance(data[field], str):
            return data[field]

    # Nếu không có, trả về empty string
    return ""


def extract_metadata_from_data(data: Dict, exclude_fields: Optional[List[str]] = None) -> Dict:
    """
    Tự động lấy TẤT CẢ metadata từ JSON data (trừ các field được exclude)

    Args:
        data: Dict chứa dữ liệu JSON
        exclude_fields: List các field cần loại bỏ (mặc định: ['text', '_source_file'])

    Returns:
        Dict chứa metadata (ví dụ: doc_id, platform, modules, layer, etc.)
    """
    if exclude_fields is None:
        exclude_fields = ['text', '_source_file',
                          '_line_number', '_item_index', '_file_size', '_format']

    metadata = {}
    for key, value in data.items():
        # Bỏ qua các field trong exclude_fields và các field bắt đầu bằng '_'
        if key not in exclude_fields and not key.startswith('_'):
            # Chỉ lấy các giá trị đơn giản (str, int, float, bool)
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            # Nếu là list rỗng hoặc list đơn giản, có thể lấy
            elif isinstance(value, list) and len(value) == 0:
                metadata[key] = []
            elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                metadata[key] = value

    return metadata
