"""Token-cost estimation from per-model pricing.

Prices are USD per million tokens and come from the model's options
(``input_mtok`` / ``output_mtok``): the built-in catalog carries them for the
Ollama Cloud models, and any model can be priced in ``config.yml``. A model
without both numbers is *unpriced* - reported as such rather than guessed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def estimate_cost(options: Dict[str, Any], prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """USD cost for one request/turn, or None if the model has no pricing."""
    inp = (options or {}).get("input_mtok")
    out = (options or {}).get("output_mtok")
    if inp is None or out is None:
        return None
    try:
        return (max(prompt_tokens, 0) * float(inp) + max(completion_tokens, 0) * float(out)) / 1_000_000
    except (TypeError, ValueError):
        return None


def format_cost(cost: Optional[float]) -> str:
    if cost is None:
        return "n/a"
    if cost == 0:
        return "$0"
    if cost < 0.0001:
        return "<$0.0001"
    return f"${cost:.4f}" if cost < 1 else f"${cost:,.2f}"


def turn_cost(provider_type: str, options: Dict[str, Any], prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Local models are free; cloud models use their configured pricing."""
    if provider_type == "local":
        return 0.0
    return estimate_cost(options, prompt_tokens, completion_tokens)
