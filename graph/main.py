"""Main entry point for AUTOSAR extractor"""
import time
from pathlib import Path

try:
    # Try relative imports first (when used as package)
    from .csv_writer import save_edges_csv, save_nodes_csv
    from .data_loader import extract_text_from_data, load_json_files, extract_metadata_from_data
    from .edge_extractor import AUTOSAREdgeExtractor
    from .node_extractor import AUTOSARNodeExtractor
    from .vllm_client import VLLMClient
except ImportError:
    # Fallback to absolute imports (when run directly)
    from csv_writer import save_edges_csv, save_nodes_csv
    from data_loader import extract_text_from_data, load_json_files, extract_metadata_from_data
    from edge_extractor import AUTOSAREdgeExtractor
    from node_extractor import AUTOSARNodeExtractor
    from vllm_client import VLLMClient


def main():
    """Main extraction pipeline"""
    # Initialize VLLM client
    client = VLLMClient()
    print("Health check:", client.health_check())

    # Initialize extractors
    node_extractor = AUTOSARNodeExtractor(client)
    edge_extractor = AUTOSAREdgeExtractor(client)

    # Data directory
    data_dir = Path("/mnt/hps/dungmt19_workspace/KBS/data")
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    print(f"Loading Json/Jsonl files from {data_dir} ... ")
    start_time = time.time()
    json_data = load_json_files(data_dir, limit=5)
    load_time = time.time() - start_time
    print(f"Loaded {len(json_data)} records in {load_time:.2f}s")

    if not json_data:
        print("No data loaded")
        return

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

    for idx, data in enumerate(json_data):
        print(f"Processing record {idx+1}/{len(json_data)}...")
        print(f"Source: {data.get('_source_file', 'unknown')}")
        text = extract_text_from_data(data)
        source_file = data.get('_source_file', 'unknown')

        if not text or len(text.strip()) < 10:
            print(f"Skipping: text too short ({len(text)} chars)")
            continue
        print(f"Text length: {len(text)} characters")

        # Extract nodes
        print("Extracting nodes...")
        node_start = time.time()

        # Tự động lấy TẤT CẢ metadata từ JSON data (doc_id, platform, modules, layer, etc.)
        metadata = extract_metadata_from_data(data)
        nodes = node_extractor.extract(
            text, file_name=source_file, metadata=metadata)
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
            edges_dict = []

        all_nodes.extend(nodes)
        save_nodes_csv(nodes, nodes_path, append=True)
        if edges_dict:
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
