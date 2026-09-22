"""One place that turns config.yml into the settings an agent carries (used by the TUI,
headless mode and background tasks, which used to each repeat this)."""

from __future__ import annotations

from typing import Any, Callable, Optional


def make_provider_builder(config_manager: Any) -> Callable[[str], Optional[Any]]:
    """A function that builds a provider from a config id (``ollama-cloud/glm-5.2``, ``claude``, ...),
    or returns None if it can't be used (unknown id, or a cloud provider with no API key)."""
    from core.providers import ModelConfig, ProviderFactory

    def build(provider_id: str) -> Optional[Any]:
        try:
            cfg = config_manager.get_provider_config(provider_id)
            ptype = cfg.get("provider_type", "cloud")
            if ptype == "cloud" and not cfg.get("api_key") and not config_manager.has_api_key(provider_id):
                return None
            return ProviderFactory.get_provider(ModelConfig(
                name=cfg.get("name", provider_id), endpoint=cfg["endpoint"], api_key=cfg.get("api_key"),
                provider_type=ptype, options=cfg.get("options", {}),
            ))
        except Exception:
            return None

    return build


def configure_agent(agent: Any, get: Callable[..., Any], config_manager: Any = None) -> Any:
    """Apply sandbox, budget, hook and failover settings from config (``get`` is ConfigManager.get or dict.get)."""
    from core.budget import Budget
    from core.hooks import Hooks
    from core.sandbox import sandbox_settings

    agent.sandbox_options = sandbox_settings(get)
    agent.sandbox_mode = agent.sandbox_options["mode"]
    agent.budget = Budget.from_config(get)
    agent.hooks = Hooks.from_config(get)
    try:
        agent.stall_timeout = max(0.0, float(get("stall_timeout", 180) or 0))  # seconds without model output; 0 = off
    except (TypeError, ValueError):
        agent.stall_timeout = 180.0
    fallbacks = get("fallback_providers", None) or []
    agent.fallback_ids = [fallbacks] if isinstance(fallbacks, str) else [str(f) for f in fallbacks]
    agent.provider_builder = make_provider_builder(config_manager) if config_manager is not None else None
    return agent
