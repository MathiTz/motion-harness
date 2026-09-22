"""Per-turn budgets: stop a turn that is using more steps, tokens, money or time than intended.

Config (all optional, per *turn*)::

    budget:
      max_steps: 12        # model calls
      max_tokens: 150000   # prompt + completion, as reported by the provider
      max_cost_usd: 0.50   # needs model pricing (input_mtok / output_mtok)
      max_seconds: 180
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from core.pricing import turn_cost


def _num(value: Any, kind: type) -> Optional[float]:
    if value in (None, "", 0, "0", False):
        return None
    try:
        n = kind(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


@dataclass
class Budget:
    max_steps: Optional[int] = None
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_seconds: Optional[float] = None

    @classmethod
    def from_config(cls, get: Callable[..., Any]) -> "Budget":
        raw = get("budget", None) or {}
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            max_steps=_num(raw.get("max_steps"), int),  # type: ignore[arg-type]
            max_tokens=_num(raw.get("max_tokens"), int),  # type: ignore[arg-type]
            max_cost_usd=_num(raw.get("max_cost_usd"), float),
            max_seconds=_num(raw.get("max_seconds"), float),
        )

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.max_steps, self.max_tokens, self.max_cost_usd, self.max_seconds))

    def exceeded(self, *, steps: int, usage: Dict[str, int], elapsed: float,
                 provider_type: str = "cloud", options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """A human-readable reason if any limit is reached, else None."""
        if self.max_steps is not None and steps >= self.max_steps:
            return f"{steps} model steps (limit {self.max_steps})"
        total = usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
        if self.max_tokens is not None and total >= self.max_tokens:
            return f"{total:,} tokens (limit {self.max_tokens:,})"
        if self.max_cost_usd is not None:
            cost = turn_cost(provider_type, options or {}, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            if cost is not None and cost >= self.max_cost_usd:
                return f"${cost:.4f} (limit ${self.max_cost_usd:.2f})"
        if self.max_seconds is not None and elapsed >= self.max_seconds:
            return f"{elapsed:.0f}s (limit {self.max_seconds:.0f}s)"
        return None

    def describe(self) -> str:
        parts = []
        if self.max_steps is not None:
            parts.append(f"{self.max_steps} steps")
        if self.max_tokens is not None:
            parts.append(f"{self.max_tokens:,} tokens")
        if self.max_cost_usd is not None:
            parts.append(f"${self.max_cost_usd:.2f}")
        if self.max_seconds is not None:
            parts.append(f"{self.max_seconds:.0f}s")
        return " · ".join(parts) if parts else "no limits"
