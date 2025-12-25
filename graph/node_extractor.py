"""Node extractor for AUTOSAR entities"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from backend.graph_extractor.prompts import AUTOSAR_ENTITY_PROMPT
from vllm_client import VLLMClient

# Add sys.path nếu cần
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))


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
