import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
import json
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    endpoint: str
    api_key: Optional[str] = None
    provider_type: str = "cloud"  # "cloud", "local", "proxy"
    options: Dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract Base Class for all model providers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=120.0)

    @abstractmethod
    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate a completion from the model."""
        pass

    async def stream_complete(self, prompt: str, system_prompt: str = "", **kwargs) -> AsyncIterator[str]:
        """Optional streaming completion. Defaults to one-shot completion."""
        result = await self.complete(prompt, system_prompt=system_prompt, **kwargs)
        if result:
            yield result

    async def close(self):
        await self._client.aclose()


class LocalProvider(BaseProvider):
    """Provider for local LLMs via Ollama-compatible /api/chat endpoint."""

    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        url = f"{self.config.endpoint.rstrip('/')}/api/chat"
        model = self.config.options.get("model", self.config.name.lower())
        payload = {
            "model": model,
            "messages": [],
            "stream": False,
            "options": {},
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        if "temperature" in self.config.options:
            payload["options"]["temperature"] = self.config.options["temperature"]
        if "num_ctx" in self.config.options:
            payload["options"]["num_ctx"] = self.config.options["num_ctx"]

        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def stream_complete(self, prompt: str, system_prompt: str = "", **kwargs) -> AsyncIterator[str]:
        url = f"{self.config.endpoint.rstrip('/')}/api/chat"
        model = self.config.options.get("model", self.config.name.lower())
        payload = {
            "model": model,
            "messages": [],
            "stream": True,
            "options": {},
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        if "temperature" in self.config.options:
            payload["options"]["temperature"] = self.config.options["temperature"]
        if "num_ctx" in self.config.options:
            payload["options"]["num_ctx"] = self.config.options["num_ctx"]

        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding via Ollama /api/embeddings."""
        url = f"{self.config.endpoint.rstrip('/')}/api/embeddings"
        model = self.config.options.get("embed_model", self.config.options.get("model", self.config.name.lower()))
        resp = await self._client.post(url, json={"model": model, "prompt": text})
        resp.raise_for_status()
        return resp.json().get("embedding", [])


class CloudProvider(BaseProvider):
    """Provider for cloud LLMs (Anthropic, OpenAI) using their native chat APIs."""

    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        endpoint = self.config.endpoint
        api_key = self.config.api_key or os.environ.get("OLLAMA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

        if "anthropic" in endpoint:
            return await self._anthropic_complete(prompt, system_prompt, api_key)
        elif "openai" in endpoint:
            return await self._openai_complete(prompt, system_prompt, api_key)
        else:
            return await self._openai_complete(prompt, system_prompt, api_key)

    async def stream_complete(self, prompt: str, system_prompt: str = "", **kwargs) -> AsyncIterator[str]:
        endpoint = self.config.endpoint
        api_key = self.config.api_key or os.environ.get("OLLAMA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

        if "anthropic" in endpoint:
            async for chunk in super().stream_complete(prompt, system_prompt=system_prompt, **kwargs):
                yield chunk
            return

        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"
        model = self.config.options.get("model", "gpt-4o")
        max_tokens = self.config.options.get("max_tokens", 4096)
        temperature = self.config.options.get("temperature", 0.8)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
        }

        async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {})
                chunk = delta.get("content", "")
                if chunk:
                    yield chunk

    async def _anthropic_complete(self, prompt: str, system_prompt: str, api_key: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        model = self.config.options.get("model", "claude-3-5-sonnet-20241022")
        max_tokens = self.config.options.get("max_tokens", 4096)
        temperature = self.config.options.get("temperature", 0.7)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", [{}])[0].get("text", "")

    async def _openai_complete(self, prompt: str, system_prompt: str, api_key: str) -> str:
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"
        model = self.config.options.get("model", "gpt-4o")
        max_tokens = self.config.options.get("max_tokens", 4096)
        temperature = self.config.options.get("temperature", 0.8)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


class ProxyProvider(BaseProvider):
    """Provider for custom proxy/gateway endpoints (OpenAI-compatible)."""

    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"
        api_key = self.config.api_key or os.environ.get("PROXY_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.options.get("model", "default"),
            "messages": messages,
            "temperature": self.config.options.get("temperature", 0.7),
        }
        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def stream_complete(self, prompt: str, system_prompt: str = "", **kwargs) -> AsyncIterator[str]:
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"
        api_key = self.config.api_key or os.environ.get("PROXY_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.options.get("model", "default"),
            "messages": messages,
            "temperature": self.config.options.get("temperature", 0.7),
            "stream": True,
        }

        async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {})
                chunk = delta.get("content", "")
                if chunk:
                    yield chunk


class ProviderFactory:
    """Factory to resolve the correct provider based on configuration."""

    @staticmethod
    def get_provider(config: ModelConfig) -> BaseProvider:
        if config.provider_type == "local":
            return LocalProvider(config)
        elif config.provider_type == "cloud":
            return CloudProvider(config)
        elif config.provider_type == "proxy":
            return ProxyProvider(config)
        else:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")