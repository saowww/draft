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

    def _merge_duplicate_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """
        Gộp các nodes trùng lặp (cùng name và semantic_type) và gộp context

        Args:
            nodes: List các node dictionaries

        Returns:
            List các nodes đã được gộp (mỗi node unique chỉ xuất hiện 1 lần)
        """
        # Nhóm nodes theo key (name + semantic_type)
        node_groups = {}
        for node in nodes:
            # Xử lý None values an toàn - chuyển thành string rỗng trước khi strip
            name = node.get('name')
            semantic_type = node.get('semantic_type')
            
            # Chuyển đổi an toàn: None -> '', sau đó strip và lower
            name_str = str(name).strip().lower() if name is not None else ''
            semantic_type_str = str(semantic_type).strip().lower() if semantic_type is not None else ''
            
            key = (name_str, semantic_type_str)

            if key not in node_groups:
                node_groups[key] = []
            node_groups[key].append(node)

        # Gộp các nodes trùng lặp
        merged_nodes = []
        for key, group in node_groups.items():
            if len(group) == 1:
                # Không trùng, giữ nguyên
                merged_nodes.append(group[0])
            else:
                # Có trùng, gộp lại
                base_node = group[0].copy()

                # Gộp tất cả context_left và context_right từ TẤT CẢ nodes trong group
                # Sử dụng 'in' check thay vì truthy check để không bỏ sót empty string
                all_context_left = []
                all_context_right = []
                for n in group:
                    # Lấy context_left nếu key tồn tại (kể cả empty string)
                    if 'context_left' in n:
                        context_left = n.get('context_left', '')
                        if context_left is not None:
                            all_context_left.append(str(context_left))
                    # Lấy context_right nếu key tồn tại (kể cả empty string)
                    if 'context_right' in n:
                        context_right = n.get('context_right', '')
                        if context_right is not None:
                            all_context_right.append(str(context_right))

                # Loại bỏ empty và duplicate, giữ lại thứ tự
                unique_left = list(dict.fromkeys(
                    [c.strip() for c in all_context_left if c and c.strip()]))
                unique_right = list(dict.fromkeys(
                    [c.strip() for c in all_context_right if c and c.strip()]))

                # Gộp context lại
                merged_context_left = ' ... '.join(unique_left) if unique_left else ''
                merged_context_right = ' ... '.join(unique_right) if unique_right else ''

                base_node['context_left'] = merged_context_left
                base_node['context_right'] = merged_context_right
                base_node['_duplicate_count'] = len(group)  # Lưu số lần trùng

                merged_nodes.append(base_node)

        return merged_nodes

    def _summarize_context_with_llm(self, context_left: str, context_right: str, entity_name: str) -> Tuple[str, str]:
        """
        Dùng LLM để rút gọn context đã gộp

        Args:
            context_left: Context bên trái đã gộp
            context_right: Context bên phải đã gộp
            entity_name: Tên của entity để LLM hiểu context

        Returns:
            Tuple (summarized_left, summarized_right)
        """
        if not context_left and not context_right:
            return "", ""

        # Tạo prompt để rút gọn context
        prompt = f"""Bạn là một chuyên gia rút gọn văn bản. Hãy rút gọn các context sau đây về entity "{entity_name}" thành một câu ngắn gọn, giữ lại thông tin quan trọng nhất.

Context bên trái (trước entity):
{context_left}

Context bên phải (sau entity):
{context_right}

Hãy trả về JSON với format:
{{
    "context_left": "câu rút gọn bên trái",
    "context_right": "câu rút gọn bên phải"
}}

Chỉ trả về JSON, không giải thích thêm."""

        try:
            resp = self.client.generate(
                user_prompt=prompt,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500
            )

            # Extract response text
            if hasattr(resp, 'message'):
                resp_text = resp.message
            elif hasattr(resp, 'choices') and resp.choices:
                resp_text = resp.choices[0].message.content
            elif hasattr(resp, 'text'):
                resp_text = resp.text
            elif isinstance(resp, str):
                resp_text = resp
            else:
                # Fallback: trả về context gốc nếu không parse được
                return context_left[:200], context_right[:200]

            # Clean markdown
            resp_clean = re.sub(r'^```[a-z]*\n?', '',
                                resp_text, flags=re.MULTILINE)
            resp_clean = re.sub(r'```$', '', resp_clean,
                                flags=re.MULTILINE).strip()

            # Parse JSON
            try:
                result = json.loads(resp_clean)
                summarized_left = result.get(
                    'context_left', context_left[:200])
                summarized_right = result.get(
                    'context_right', context_right[:200])
                return summarized_left, summarized_right
            except json.JSONDecodeError:
                # Fallback: rút gọn thủ công nếu LLM không trả về JSON
                return context_left[:200], context_right[:200]

        except Exception as e:
            print(f"Error summarizing context with LLM: {e}")
            # Fallback: trả về context gốc đã rút ngắn
            return context_left[:200], context_right[:200]

    def extract(self, text: str, file_name: str = 'unknown', metadata: Dict = None) -> List[Dict]:
        """Extract AUTOSAR entities from text.

        Args:
            text: Text to extract entities from
            file_name: Source file name
            metadata: Optional metadata dictionary with fields like doc_id, platform, modules
        """
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
                    node_dict = {
                        "name": name,
                        "semantic_type": semantic_type,
                        "mention": mention,
                        "context_left": context_left,
                        "context_right": context_right,
                        "source_file": file_name,
                    }

                    # Tự động thêm TẤT CẢ metadata fields từ JSON vào node
                    if metadata:
                        node_dict.update(metadata)

                    output.append(node_dict)

            # Xử lý nodes trùng lặp: gộp context và rút gọn bằng LLM
            if output:
                print(f"  Trước khi gộp: {len(output)} nodes")
                merged_nodes = self._merge_duplicate_nodes(output)
                print(f"  Sau khi gộp: {len(merged_nodes)} nodes")

                # Rút gọn context của các nodes đã gộp bằng LLM
                final_nodes = []
                for node in merged_nodes:
                    if node.get('_duplicate_count', 1) > 1:
                        # Node đã được gộp, cần rút gọn context
                        print(
                            f"  Đang rút gọn context cho node: {node.get('name')} ({node.get('_duplicate_count')} lần trùng)")
                        summarized_left, summarized_right = self._summarize_context_with_llm(
                            node.get('context_left', ''),
                            node.get('context_right', ''),
                            node.get('name', '')
                        )
                        node['context_left'] = summarized_left
                        node['context_right'] = summarized_right
                        # Xóa field tạm
                        node.pop('_duplicate_count', None)
                    final_nodes.append(node)

                return final_nodes

            return output

        except Exception as e:
            print(f"Error extracting entities: {e}")
            import traceback
            traceback.print_exc()
            return []
