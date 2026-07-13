import re
from typing import List, Optional

class CavemanCompressor:
    """
    Implements the 'Caveman' token compression protocol.
    Reduces token usage by stripping conversational fluff, politeness, and 
    redundant structures while preserving technical precision.
    """

    # Patterns to strip from internal agent-to-agent or tool-to-agent communication
    FLUFF_PATTERNS = [
        (r"(?i)^(I'm sorry, )", ""),
        (r"(?i)^(I apologize, )", ""),
        (r"(?i)^(Certainly! )", ""),
        (r"(?i)^(Of course! )", ""),
        (r"(?i)^(Here is the result: )", ""),
        (r"(?i)^(Based on the provided context, )", ""),
        (r"(?i)^(I have analyzed the files and found that )", ""),
        (r"(?i)( please let me know if you need further assistance\.)", ""),
        (r"(?i)( I hope this helps\.)", ""),
        (r"(?i)( Thank you\.)", ""),
        (r"(?i)( I'm sorry for the inconvenience\.)", ""),
        (r"(?i)( I apologize for the inconvenience\.)", ""),
    ]

    @classmethod
    def compress(cls, text: str) -> str:
        """
        Compresses text into 'Caveman' mode.
        Usage: Used for internal communication between agents or tool outputs.
        """
        if not text:
            return ""

        compressed = text
        for pattern, replacement in cls.FLUFF_PATTERNS:
            compressed = re.sub(pattern, replacement, compressed)

        # Remove excessive whitespace
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        return compressed

    @classmethod
    def expand(cls, text: str) -> str:
        """
        Optional: Expands Caveman text back into polite prose for the end user.
        """
        # Currently, we keep the internal 'Caveman' style for efficiency.
        # This can be expanded with a specialized LLM prompt if needed.
        return text

class CavemanProtocol:
    """
    Orchestrates when to use compression based on the communication channel.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def process_outgoing(self, text: str, target: str = "agent") -> str:
        """
        Process outgoing messages. If target is another agent or a tool, 
        compress the output.
        """
        if self.enabled and target != "user":
            return CavemanCompressor.compress(text)
        return text

    def process_incoming(self, text: str, source: str = "agent") -> str:
        """
        Process incoming messages.
        """
        # We typically keep the compressed form to save prompt tokens.
        return text
