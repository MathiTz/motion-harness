import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    endpoint: str
    api_key: Optional[str] = None
    provider_type: str = "cloud"  # "cloud", "local", "proxy"
    options: Dict[str, Any] = None

class BaseProvider(ABC):
    """Abstract Base Class for all model providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate a completion from the model."""
        pass

class LocalProvider(BaseProvider):
    """Provider for local LLMs (e.g., Ollama, vLLM)."""
    
    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        # Implementation for local HTTP calls to Ollama/vLLM
        # In a real scenario, this would use httpx or aiohttp
        return f"[Local Model {self.config.name}] Response to: {prompt[:20]}..."

class CloudProvider(BaseProvider):
    """Provider for cloud LLMs (e.g., Anthropic, OpenAI)."""
    
    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        # Implementation for cloud SDKs/APIs
        return f"[Cloud Model {self.config.name}] Response to: {prompt[:20]}..."

class ProviderFactory:
    """Factory to resolve the correct provider based on configuration."""
    
    @staticmethod
    def get_provider(config: ModelConfig) -> BaseProvider:
        if config.provider_type == "local":
            return LocalProvider(config)
        elif config.provider_type == "cloud":
            return CloudProvider(config)
        else:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")
