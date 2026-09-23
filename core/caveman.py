import re
from typing import List, Optional, Tuple


class CavemanCompressor:
    """
    Implements the 'Caveman' token compression protocol.
    Reduces token usage by stripping conversational fluff, politeness, and 
    redundant structures while preserving technical precision.

    Bidirectional: compress() tracks what was removed so expand() can restore it.
    """

    FLUFF_PATTERNS: List[Tuple[str, str]] = [
        (r"(?i)^(I'm sorry, )", "greeting-sorry"),
        (r"(?i)^(I apologize, )", "greeting-apologize"),
        (r"(?i)^(Certainly! )", "greeting-certainly"),
        (r"(?i)^(Of course! )", "greeting-ofcourse"),
        (r"(?i)^(Here is the result: )", "prefix-result"),
        (r"(?i)^(Based on the provided context, )", "prefix-context"),
        (r"(?i)^(I have analyzed the files and found that )", "prefix-analyzed"),
        (r"(?i)( please let me know if you need further assistance\.)", "closing-assist"),
        (r"(?i)( I hope this helps\.)", "closing-hope"),
        (r"(?i)( Thank you\.)", "closing-thanks"),
        (r"(?i)( I'm sorry for the inconvenience\.)", "closing-inconvenience-sorry"),
        (r"(?i)( I apologize for the inconvenience\.)", "closing-inconvenience-apologize"),
    ]

    @classmethod
    def compress(cls, text: str) -> Tuple[str, List[Tuple[int, str, str]]]:
        """
        Compresses text into 'Caveman' mode: splices every FLUFF_PATTERNS match out of ``text``,
        in the order the matches actually occur (not FLUFF_PATTERNS' list order - two closing
        phrases in one response must come back in the order they were written, not the order this
        list happens to check them in).

        Returns (compressed_text, fragments). Each fragment is (position, tag, original_text),
        where ``position`` is the exact offset in ``compressed_text`` at which ``original_text``
        must be reinserted to reconstruct the input exactly - see expand(). No whitespace beyond
        each match's own text is ever touched: every FLUFF_PATTERNS entry already includes its own
        delimiting space inside the match (e.g. "Certainly! " with the trailing space, " Thank you."
        with the leading space), so removing a match cleanly leaves no extra whitespace behind.
        Anything between/around matches - including newlines, indentation, or a code block - is
        copied through untouched, which is what makes expand() an exact inverse for any input, not
        just single-line prose.
        """
        if not text:
            return "", []

        matches = []
        for pattern, tag in cls.FLUFF_PATTERNS:
            for m in re.finditer(pattern, text):
                matches.append((m.start(), m.end(), tag, m.group(0)))
        matches.sort(key=lambda m: m[0])

        # Two FLUFF_PATTERNS entries are never expected to overlap on real model output, but a
        # pathological input could still match two at the same spot - keep whichever match starts
        # first (matches() is stable-sorted by start above) and drop anything that overlaps it,
        # rather than splicing out overlapping ranges incorrectly.
        filtered = []
        last_end = -1
        for start, end, tag, original in matches:
            if start < last_end:
                continue
            filtered.append((start, end, tag, original))
            last_end = end

        fragments: List[Tuple[int, str, str]] = []
        parts: List[str] = []
        cursor = 0
        compressed_len = 0
        for start, end, tag, original in filtered:
            piece = text[cursor:start]
            parts.append(piece)
            compressed_len += len(piece)
            fragments.append((compressed_len, tag, original))
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts), fragments

    @classmethod
    def expand(cls, text: str, fragments: List[Tuple[int, str, str]]) -> str:
        """
        Restores compressed text to the original by reinserting each fragment at the exact
        position compress() recorded for it - a true inverse of compress() for any input,
        including multi-line/indented text.

        Two removed fragments with nothing between them in the original (e.g. two closing phrases
        back to back) get the SAME recorded position - compress() had no text to advance the
        cursor over between them. They must be reinserted together, in their original order, not
        one at a time: splicing the first back in, then splicing the second in at that same
        position, would push the second one ahead of the first. ``fragments`` is already in
        original-text order (compress() appends in the order matches occur), so grouping
        consecutive equal positions and joining their text is enough to restore that order.
        Groups are then applied from the last position to the first so inserting one never shifts
        the recorded position of a group still waiting to be applied.
        """
        if not fragments:
            return text

        groups: List[Tuple[int, str]] = []
        for position, _tag, original in fragments:
            if groups and groups[-1][0] == position:
                groups[-1] = (position, groups[-1][1] + original)
            else:
                groups.append((position, original))

        result = text
        for position, combined in sorted(groups, key=lambda g: g[0], reverse=True):
            result = result[:position] + combined + result[position:]
        return result


class CavemanProtocol:
    """
    Orchestrates when to use compression based on the communication channel.
    Tracks fragments from compress() so that expand() can reverse it.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._fragments: List[Tuple[int, str, str]] = []

    def process_outgoing(self, text: str, target: str = "agent") -> str:
        """
        Process outgoing messages. If target is another agent or a tool, 
        compress the output and track fragments for decompression.
        """
        if self.enabled and target != "user":
            compressed, fragments = CavemanCompressor.compress(text)
            self._fragments = fragments
            return compressed
        return text

    def process_incoming(self, text: str, source: str = "agent") -> str:
        """
        Process incoming messages. Reconstruct natural language if fragments
        are available from a prior compression.
        """
        if self._fragments:
            expanded = CavemanCompressor.expand(text, self._fragments)
            self._fragments = []  # Consume fragments after expansion
            return expanded
        return text
