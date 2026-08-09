"""Tests for local model auto-discovery (Ollama /api/tags) and the
Ollama Cloud model scraper."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.catalog import (
    BUILTIN_CATALOG,
    discover_local_models,
    scrape_ollama_cloud_models,
    update_ollama_cloud_models,
)
from core.providers import LocalProvider, ModelConfig


def _fake_tags_response(models):
    resp = MagicMock()
    resp.json.return_value = {"models": [{"name": m} for m in models]}
    return resp


async def test_discover_local_models_returns_installed():
    cfg = {
        "provider_type": "local",
        "endpoint": "http://localhost:11434",
        "options": {"model": "llama3"},
    }
    fake_resp = _fake_tags_response(["llama3:latest", "qwen3:8b", "nomic-embed-text:latest"])

    with patch("core.providers.httpx.AsyncClient") as MockClient:
        client = MockClient.return_value
        client.get = AsyncMock(return_value=fake_resp)
        client.aclose = AsyncMock()
        models = await discover_local_models("local-llama", cfg)

    assert "llama3:latest" in models
    assert "qwen3:8b" in models
    assert "nomic-embed-text:latest" in models


async def test_discover_local_models_skips_non_local():
    cfg = {"provider_type": "cloud", "endpoint": "https://api.openai.com/v1"}
    models = await discover_local_models("gpt-4o", cfg)
    assert models == []


async def test_discover_local_models_falls_back_on_error():
    cfg = {
        "provider_type": "local",
        "endpoint": "http://localhost:11434",
        "options": {"model": "llama3"},
    }
    with patch("core.providers.httpx.AsyncClient") as MockClient:
        client = MockClient.return_value
        client.get.side_effect = Exception("connection refused")
        client.aclose = AsyncMock()
        models = await discover_local_models("local-llama", cfg)

    # Falls back to the configured model.
    assert models == ["llama3"]


async def test_local_provider_list_models_uses_tags_endpoint():
    provider = LocalProvider(ModelConfig(
        name="local-llama",
        endpoint="http://localhost:11434",
        provider_type="local",
        options={"model": "llama3"},
    ))
    fake_resp = _fake_tags_response(["llama3:latest"])

    with patch.object(provider._client, "get", return_value=fake_resp) as mock_get:
        models = await provider.list_models()

    assert models == ["llama3:latest"]
    mock_get.assert_called_once_with("http://localhost:11434/api/tags")


# ─── Ollama Cloud scraper ────────────────────────────────────────────────────

_HTML = """
<html><body>
<a href="/library/glm-5.2">GLM-5.2</a>
<a href="/library/deepseek-v4-flash">DeepSeek V4 Flash</a>
<a href="/search/qwen3.5">Qwen 3.5</a>
<a href="/library/glm-5.2">duplicate</a>
</body></html>
"""


async def test_scrape_ollama_cloud_models_extracts_names():
    fake_resp = MagicMock()
    fake_resp.text = _HTML

    with patch("core.catalog.httpx.AsyncClient") as MockClient:
        client = MockClient.return_value
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        models = await scrape_ollama_cloud_models()

    assert "glm-5.2" in models
    assert "deepseek-v4-flash" in models
    assert "qwen3.5" in models
    # Duplicates are removed.
    assert models.count("glm-5.2") == 1


async def test_scrape_ollama_cloud_models_returns_empty_on_error():
    with patch("core.catalog.httpx.AsyncClient") as MockClient:
        client = MockClient.return_value
        client.get.side_effect = Exception("network error")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        models = await scrape_ollama_cloud_models()
    assert models == []


async def test_update_ollama_cloud_models_merges_into_catalog():
    fake_resp = MagicMock()
    fake_resp.text = '<a href="/library/brand-new-model">New</a>'

    with patch("core.catalog.httpx.AsyncClient") as MockClient:
        client = MockClient.return_value
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        added = await update_ollama_cloud_models()

    assert added >= 1
    assert "brand-new-model" in BUILTIN_CATALOG["ollama-cloud"]["models"]
