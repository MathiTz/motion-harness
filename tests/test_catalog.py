"""Tests for the built-in provider/model catalog."""
from core.catalog import BUILTIN_CATALOG, merge_catalog


def test_catalog_has_expected_providers():
    assert "ollama-cloud" in BUILTIN_CATALOG
    assert "claude" in BUILTIN_CATALOG
    assert "openai" in BUILTIN_CATALOG
    assert "local-llama" in BUILTIN_CATALOG


def test_catalog_ollama_has_models():
    models = BUILTIN_CATALOG["ollama-cloud"]["models"]
    assert "deepseek-v4-flash" in models
    assert "qwen3-coder:480b" in models


def test_catalog_claude_has_many_models():
    models = BUILTIN_CATALOG["claude"]["models"]
    assert "claude-opus-5" in models
    assert "claude-sonnet-5" in models
    assert "claude-haiku-4-5" in models
    assert len(models) >= 8


def test_catalog_openai_has_many_models():
    models = BUILTIN_CATALOG["openai"]["models"]
    assert "gpt-5.6-sol" in models
    assert "gpt-4o" in models
    assert "o3" in models
    assert len(models) >= 15


def test_merge_catalog_fills_missing_providers():
    merged = merge_catalog({})
    # Catalog providers are present even with an empty user config.
    assert "openai" in merged
    assert "claude" in merged


def test_merge_catalog_user_overrides_catalog():
    user = {
        "gpt-4o": {
            "name": "My GPT",
            "options": {"model": "gpt-4o-mini", "temperature": 0.1},
        }
    }
    merged = merge_catalog(user)
    # User name wins.
    assert merged["gpt-4o"]["name"] == "My GPT"
    # User options override catalog defaults.
    assert merged["gpt-4o"]["options"]["model"] == "gpt-4o-mini"
    assert merged["gpt-4o"]["options"]["temperature"] == 0.1


def test_merge_catalog_preserves_default_key():
    user = {"default": "gpt-4o"}
    merged = merge_catalog(user)
    # The default key is not treated as a provider.
    assert "default" not in merged


def test_merge_catalog_deep_merges_models():
    user = {
        "ollama-cloud": {
            "models": {"custom-model": {"temperature": 0.3}},
        }
    }
    merged = merge_catalog(user)
    # Catalog models preserved, user model added.
    assert "deepseek-v4-flash" in merged["ollama-cloud"]["models"]
    assert "custom-model" in merged["ollama-cloud"]["models"]
