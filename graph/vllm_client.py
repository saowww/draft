from typing import Optional, Dict, Any, List
from openai import OpenAI
from ..base.llm_client import BaseLLMClient, LLMResponse  # Nếu có base classes
from .vllm_config import VLLMConfig


class VLLMClient(BaseLLMClient):
    def __init__(self, config: VLLMConfig = None):
        if config is None:
            config = VLLMConfig.from_yaml("/mnt/hps/dungmt19_workspace/KBS/backend/llm/config.yaml")
        try:
            super().__init__(config)
        except TypeError:
            self.config = config
        
        self.config: VLLMConfig = config
        
        self.chat_client = OpenAI(
            api_key=config.api_key, 
            base_url=config.chat_base_url
        )
        self.embedding_client = OpenAI(
            api_key=config.api_key, 
            base_url=config.embedding_base_url
        )
    
    def generate(self, 
                 user_prompt: str, 
                 system_prompt: Optional[str] = None, 
                 **kwargs) -> LLMResponse:
        """Generate text with full system/user messages"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Merge config + override kwargs
        gen_config = {**self.config.get_chat_params(), **kwargs}
        
        response = self.chat_client.chat.completions.create(
            model=self.config.chat_model,  # GLM-4.6-AWQ
            messages=messages,
            **gen_config
        )

        message = response.choices[0].message.content or ""
        if not message:
            raise RuntimeError("vLLM response invalid or missing content")

        # Full metadata usage stats
        metadata = {
            "model": self.config.chat_model,
            "model_id": getattr(response, 'model', ''),
            "usage": {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0),
            }
        }

        return LLMResponse(message=message, metadata=metadata)
    
    def embed(self, text: str) -> List[float]:
        """Embedding with Qwen"""
        response = self.embedding_client.embeddings.create(
            input=[text],  # List[str] required
            model=self.config.embedding_model,  # qwen3-embedding-0.6B
            **self.config.get_embedding_params()
        )
        
        embedding = response.data[0].embedding
        if not embedding:
            raise ValueError("Embedding not found in vLLM response")
        
        return embedding
    
    def list_models(self) -> List[Dict[str, str]]:
        """List chat models (GLM server)"""
        try:
            response = self.chat_client.models.list()
            return [{"id": m.id, "object": m.object} for m in response.data]
        except Exception as e:
            print(f"Chat models list error: {e}")
            return []
    
    def list_embedding_models(self) -> List[Dict[str, str]]:
        """List embedding models (Qwen server)"""
        try:
            response = self.embedding_client.models.list()
            return [{"id": m.id, "object": m.object} for m in response.data]
        except Exception as e:
            print(f"Embedding models list error: {e}")
            return []
    
    def health_check(self) -> Dict[str, bool]:
        """Check 2 servers health"""
        try:
            self.chat_client.models.list()
            chat_ok = True
        except:
            chat_ok = False
        
        try:
            self.embedding_client.models.list()
            embed_ok = True
        except:
            embed_ok = False
            
        return {
            "chat_server": chat_ok,
            "embedding_server": embed_ok,
            "chat_url": self.config.chat_base_url,
            "embed_url": self.config.embedding_base_url
        }
    
    def __repr__(self):
        return (f"VLLMClient(chat={self.config.chat_model}@{self.config.chat_base_url}, "
                f"embed={self.config.embedding_model}@{self.config.embedding_base_url})")
