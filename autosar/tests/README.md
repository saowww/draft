# Test Cases cho AUTOSAR Extractor

Thư mục này chứa các test cases cho từng module trong package autosar.

## Cấu trúc Test Files

- `test_data_loader.py` - Test cases cho `data_loader.py`
  - `load_json_files()` - Load JSON/JSONL files
  - `extract_text_from_data()` - Extract text from data

- `test_node_extractor.py` - Test cases cho `node_extractor.py`
  - `AUTOSARNodeExtractor` class
  - `extract()` method
  - `extract_context()` method

- `test_edge_extractor.py` - Test cases cho `edge_extractor.py`
  - `AUTOSAREdgeExtractor` class
  - `extract()` method

- `test_csv_writer.py` - Test cases cho `csv_writer.py`
  - `save_nodes_csv()` - Save nodes to CSV
  - `save_edges_csv()` - Save edges to CSV

- `test_vllm_client.py` - Test cases cho `vllm_client.py`
  - `VLLMClient` class
  - `generate()` method
  - `embed()` method
  - `health_check()` method

- `test_main.py` - Test cases cho `main.py`
  - `main()` function
  - Integration tests

## Cách chạy tests

### Sử dụng unittest (Python built-in):
```bash
cd autosar
python -m unittest discover tests
```

### Chạy một test file cụ thể:
```bash
python -m unittest tests.test_data_loader
```

### Chạy một test case cụ thể:
```bash
python -m unittest tests.test_data_loader.TestDataLoader.test_load_json_files_single_json
```

### Sử dụng pytest (nếu đã cài đặt):
```bash
pytest tests/
```

### Chạy với coverage:
```bash
pytest tests/ --cov=autosar --cov-report=html
```

## Test Coverage

Các test cases bao gồm:
- ✅ Normal cases (happy path)
- ✅ Edge cases (empty input, invalid data)
- ✅ Error handling (exceptions, invalid JSON)
- ✅ Different response formats (LLMResponse, OpenAI format, string)
- ✅ File operations (CSV writing, JSON loading)
- ✅ Integration tests

## Notes

- Một số tests sử dụng `unittest.mock` để mock external dependencies
- Tests sử dụng `tempfile` để tạo temporary files/directories
- Tests được thiết kế để chạy độc lập, không cần external services

