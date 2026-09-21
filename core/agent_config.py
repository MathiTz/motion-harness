"""One place that turns config.yml into the settings an agent carries (used by the TUI,
headless mode and background tasks, which used to each repeat this)."""

from __future__ import annotations

from typing import Any, Callable


def configure_agent(agent: Any, get: Callable[..., Any]) -> Any:
    """Apply sandbox, budget and hook settings from config (``get`` is ConfigManager.get or dict.get)."""
    from core.budget import Budget
    from core.hooks import Hooks
    from core.sandbox import sandbox_settings

    agent.sandbox_options = sandbox_settings(get)
    agent.sandbox_mode = agent.sandbox_options["mode"]
    agent.budget = Budget.from_config(get)
    agent.hooks = Hooks.from_config(get)
    return agent
