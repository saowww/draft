"""Edge extractor for AUTOSAR relationships"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

from backend.graph_extractor.prompts import AUTOSAR_EDGE_PROMPT
from backend.graph_extractor.schema import ExtractedEdges
from vllm_client import VLLMClient

# Add sys.path nếu cần
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))


class AUTOSAREdgeExtractor:
    """Extract AUTOSAR relationships (edges) from text using entities"""

    def __init__(self, client: VLLMClient):
        self.client = client

    def extract(self, text: str, nodes: List[Dict], file_name: str = "unknown") -> ExtractedEdges:
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
                return ExtractedEdges(edges=[])

            # Clean markdown code fences
            resp_clean = re.sub(r'^```[a-z]*\n?', '',  # fix
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
            print(f"Error extracting edges: {e}")
            import traceback
            traceback.print_exc()
            return ExtractedEdges(edges=[])
