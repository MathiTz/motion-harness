"""Built-in provider/model catalog.

Mirrors opencode's approach: ship a pre-configured catalog of providers and
models so users only need to add API keys (or point at a local endpoint),
instead of configuring every model by hand.

The catalog is merged with the user's ``config.yml``: user-defined providers
override catalog entries with the same id, and catalog providers are always
available as a fallback.
"""

from typing import Any, Dict, List, Optional

import httpx
import re

from core.providers import LocalProvider, ModelConfig

# ── Built-in catalog ─────────────────────────────────────────────────────────
# Each provider: id -> {name, endpoint, provider_type, models: {model: opts}}
# ``api_key`` is intentionally omitted — it comes from config.yml or env vars.

BUILTIN_CATALOG: Dict[str, Dict[str, Any]] = {
    "ollama-cloud": {
        "name": "Ollama Cloud",
        "endpoint": "https://ollama.com/v1",
        "provider_type": "cloud",
        "default_model": "deepseek-v4-flash",
        "models": {
            "deepseek-v4-flash": {"temperature": 0.7, "max_tokens": 4096},
            "qwen3-coder:480b": {"temperature": 0.5, "max_tokens": 8192},
            "nematron-3-super": {"temperature": 0.7, "max_tokens": 4096},
            "glm-5.2": {"temperature": 0.7, "max_tokens": 4096},
            "gemma4:31b": {"temperature": 0.7, "max_tokens": 4096},
            "qwen3.5:397b": {"temperature": 0.7, "max_tokens": 4096},
            "glm-5.1": {"temperature": 0.7, "max_tokens": 4096},
            "minimax-m2.7": {"temperature": 0.7, "max_tokens": 4096},
        },
    },
    "claude": {
        "name": "Claude (Anthropic)",
        "endpoint": "https://api.anthropic.com",
        "provider_type": "cloud",
        "default_model": "claude-opus-5",
        "models": {
            "claude-opus-5": {"temperature": 0.7, "max_tokens": 128000},
            "claude-sonnet-5": {"temperature": 0.7, "max_tokens": 128000},
            "claude-haiku-4-5": {"temperature": 0.7, "max_tokens": 64000},
            "claude-fable-5": {"temperature": 0.7, "max_tokens": 128000},
            "claude-opus-4-8": {"temperature": 0.7, "max_tokens": 128000},
            "claude-opus-4-7": {"temperature": 0.7, "max_tokens": 128000},
            "claude-opus-4-6": {"temperature": 0.7, "max_tokens": 128000},
            "claude-sonnet-4-6": {"temperature": 0.7, "max_tokens": 128000},
            "claude-sonnet-4-5": {"temperature": 0.7, "max_tokens": 64000},
            "claude-opus-4-5": {"temperature": 0.7, "max_tokens": 64000},
        },
    },
    "openai": {
        "name": "OpenAI",
        "endpoint": "https://api.openai.com/v1",
        "provider_type": "cloud",
        "default_model": "gpt-5.6-sol",
        "models": {
            "gpt-5.6-sol": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.6-terra": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.6-luna": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.5": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.4": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.4-mini": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.4-nano": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.3": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.2": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5.1": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5-mini": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-5-nano": {"temperature": 0.8, "max_tokens": 128000},
            "gpt-4.1": {"temperature": 0.8, "max_tokens": 32000},
            "gpt-4.1-mini": {"temperature": 0.8, "max_tokens": 32000},
            "gpt-4.1-nano": {"temperature": 0.8, "max_tokens": 32000},
            "gpt-4o": {"temperature": 0.8, "max_tokens": 16000},
            "gpt-4o-mini": {"temperature": 0.8, "max_tokens": 16000},
            "o3": {"temperature": 0.8, "max_tokens": 100000},
            "o3-mini": {"temperature": 0.8, "max_tokens": 100000},
            "o4-mini": {"temperature": 0.8, "max_tokens": 100000},
            "o1": {"temperature": 0.8, "max_tokens": 100000},
            "o1-pro": {"temperature": 0.8, "max_tokens": 100000},
        },
    },
    "local-llama": {
        "name": "Llama 3 (Local)",
        "endpoint": "http://localhost:11434",
        "provider_type": "local",
        "options": {
            "model": "llama3",
            "temperature": 0.6,
            "embed_model": "nomic-embed-text",
        },
    },
}


def merge_catalog(user_providers: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the built-in catalog with user-defined providers.

    User providers win on id collision; catalog providers fill in anything the
    user hasn't defined. The ``default`` key (if present) is preserved.
    """
    merged: Dict[str, Any] = {}
    # Start with the catalog.
    for pid, cfg in BUILTIN_CATALOG.items():
        merged[pid] = dict(cfg)
    # Overlay user providers.
    for pid, cfg in (user_providers or {}).items():
        if pid == "default":
            continue
        if pid in merged and isinstance(cfg, dict):
            # Deep-merge: user options/models override catalog defaults.
            base = merged[pid]
            for key, value in cfg.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    base[key] = {**base[key], **value}
                else:
                    base[key] = value
        else:
            merged[pid] = cfg
    return merged


async def discover_local_models(provider_id: str, cfg: Dict[str, Any]) -> List[str]:
    """Query a local provider for installed models (Ollama /api/tags).

    Returns the list of model names, or an empty list if the provider is not
    local or the endpoint is unreachable.
    """
    if cfg.get("provider_type") != "local":
        return []
    endpoint = cfg.get("endpoint", "")
    if not endpoint:
        return []
    model_config = ModelConfig(
        name=provider_id,
        endpoint=endpoint,
        provider_type="local",
        options=cfg.get("options", {}),
    )
    provider = LocalProvider(model_config)
    try:
        return await provider.list_models()
    finally:
        await provider.close()


# ── Ollama Cloud model scraper ────────────────────────────────────────────────

OLLAMA_SEARCH_URL = "https://ollama.com/search"


async def scrape_ollama_cloud_models() -> List[str]:
    """Fetch the list of models available on Ollama Cloud.

    Scrapes the ollama.com search page and extracts model names. Returns a
    list of model ids (e.g. ``deepseek-v4-flash``, ``glm-5.2``). Returns an
    empty list if the fetch or parse fails.
    """
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(OLLAMA_SEARCH_URL)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return []

    # Model names appear as links to /library/<name> or /search/<name>.
    # Match the model name tokens in the page.
    names: List[str] = []
    seen = set()
    # Pattern for model links: href="/library/<name>" or "/search/<name>"
    for m in re.finditer(r'href="/(?:library|search)/([a-z0-9][a-z0-9._-]*)"', html):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


async def update_ollama_cloud_models() -> int:
    """Scrape Ollama Cloud models and merge them into the built-in catalog.

    Returns the number of models added/updated. The catalog is updated in
    memory (BUILTIN_CATALOG) so the model dialog reflects the latest list.
    """
    names = await scrape_ollama_cloud_models()
    if not names:
        return 0
    provider = BUILTIN_CATALOG.get("ollama-cloud", {})
    models = provider.setdefault("models", {})
    added = 0
    for name in names:
        if name not in models:
            models[name] = {"temperature": 0.7, "max_tokens": 4096}
            added += 1
    return added
