"""Bound text sent to the model by keeping its start AND its end.

Command output, logs and stack traces put the useful part (the error, the summary line, the exit
status) at the end, so a plain ``text[:limit]`` cut throws away exactly what the model needs. These
helpers keep a small head for context and a larger tail, with an explicit marker saying how much was
left out in between. The result never exceeds ``limit`` characters.
"""

from __future__ import annotations

HEAD_FRACTION = 0.25


def _marker(omitted: int) -> str:
    return f"\n…[{omitted} characters omitted from the middle; narrow the command (grep/head/tail) to see them]…\n"


def bound_text(text: str, limit: int, head_fraction: float = HEAD_FRACTION) -> str:
    """``text`` unchanged if it fits, else its start and end with an omission marker between."""
    if len(text) <= limit:
        return text
    budget = limit - len(_marker(len(text)))
    if budget <= 0:
        return text[:limit]
    head_n = int(budget * head_fraction)
    tail_n = budget - head_n
    return text[:head_n] + _marker(len(text) - head_n - tail_n) + (text[-tail_n:] if tail_n else "")


class OutputCapture:
    """Streaming version of ``bound_text``: feed chunks as they arrive, memory stays bounded by
    ``limit`` however much is written (``yes`` piped to a process must not fill RAM)."""

    def __init__(self, limit: int, head_fraction: float = HEAD_FRACTION) -> None:
        self.limit = limit
        self.head_fraction = head_fraction
        self._head_max = int(limit * head_fraction)
        self._tail_max = limit - self._head_max
        self._head = ""
        self._tail = ""
        self.total = 0

    def feed(self, text: str) -> None:
        self.total += len(text)
        if len(self._head) < self._head_max:
            take = text[: self._head_max - len(self._head)]
            self._head += take
            text = text[len(take):]
        if text:
            self._tail = (self._tail + text)[-self._tail_max:]

    @property
    def truncated(self) -> bool:
        return self.total > self.limit

    def render(self) -> str:
        if not self.truncated:
            return self._head + self._tail
        budget = self.limit - len(_marker(self.total))
        if budget <= 0:
            return self._head[: self.limit]
        head_n = min(len(self._head), int(budget * self.head_fraction))
        tail_n = budget - head_n
        return self._head[:head_n] + _marker(self.total - head_n - tail_n) + (self._tail[-tail_n:] if tail_n else "")
