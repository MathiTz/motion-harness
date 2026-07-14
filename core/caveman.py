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
    def compress(cls, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Compresses text into 'Caveman' mode.
        Returns (compressed_text, fragments) where fragments tracks each
        stripped piece and its tag for reversible decompression.
        """
        if not text:
            return "", []

        fragments: List[Tuple[str, str]] = []
        compressed = text
        for pattern, tag in cls.FLUFF_PATTERNS:
            for match in re.finditer(pattern, compressed):
                fragments.append((tag, match.group(0)))
            compressed = re.sub(pattern, "", compressed)

        # Collapse excessive whitespace left by removals
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        return compressed, fragments

    @classmethod
    def expand(cls, text: str, fragments: List[Tuple[str, str]]) -> str:
        """
        Restores compressed text to natural language by reinserting
        the tracked fragments in their original positions.
        """
        if not fragments:
            return text

        result = text
        # Reinsert prefix-type fragments at the start
        prefix_tags = {"greeting-sorry", "greeting-apologize", "greeting-certainly",
                       "greeting-ofcourse", "prefix-result", "prefix-context",
                       "prefix-analyzed"}
        # Reinsert closing-type fragments at the end
        closing_tags = {"closing-assist", "closing-hope", "closing-thanks",
                        "closing-inconvenience-sorry", "closing-inconvenience-apologize"}

        prefix_parts = []
        closing_parts = []
        for tag, original in fragments:
            if tag in prefix_tags:
                prefix_parts.append(original)
            elif tag in closing_tags:
                closing_parts.append(original)

        if prefix_parts:
            result = ''.join(prefix_parts) + result
        if closing_parts:
            result = result + ' '.join(closing_parts)

        return result


class CavemanProtocol:
    """
    Orchestrates when to use compression based on the communication channel.
    Tracks fragments from compress() so that expand() can reverse it.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._fragments: List[Tuple[str, str]] = []

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
