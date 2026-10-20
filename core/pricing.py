"""Token-cost estimation from per-model pricing.

Prices are USD per million tokens and come from the model's options
(``input_mtok`` / ``output_mtok``, and optionally ``cached_mtok`` for prompt tokens the provider
served from its cache): the built-in catalog carries them for the
Ollama Cloud models, and any model can be priced in ``config.yml``. A model
without both numbers is *unpriced* - reported as such rather than guessed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def estimate_cost(options: Dict[str, Any], prompt_tokens: int, completion_tokens: int,
                  cached_tokens: int = 0) -> Optional[float]:
    """USD cost for one request/turn, or None if the model has no pricing.

    `cached_tokens` (already part of `prompt_tokens`) are priced at the model's optional
    `cached_mtok` when it is set. When it is not, they are priced as ordinary input: that is an
    upper bound, never a guess at a discount nobody told us about."""
    inp = (options or {}).get("input_mtok")
    out = (options or {}).get("output_mtok")
    if inp is None or out is None:
        return None
    try:
        prompt = max(prompt_tokens, 0)
        cached = min(max(cached_tokens, 0), prompt)
        cached_price = (options or {}).get("cached_mtok")
        if cached and cached_price is not None:
            input_cost = (prompt - cached) * float(inp) + cached * float(cached_price)
        else:
            input_cost = prompt * float(inp)
        return (input_cost + max(completion_tokens, 0) * float(out)) / 1_000_000
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


def turn_cost(provider_type: str, options: Dict[str, Any], prompt_tokens: int, completion_tokens: int,
              cached_tokens: int = 0) -> Optional[float]:
    """Local models are free; cloud models use their configured pricing."""
    if provider_type == "local":
        return 0.0
    return estimate_cost(options, prompt_tokens, completion_tokens, cached_tokens)
