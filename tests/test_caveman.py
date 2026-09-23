"""CavemanCompressor round-trip correctness.

Regression coverage for two real bugs found while verifying README claims against the actual code
(see issue #12): (1) compress() used to blanket-collapse ALL whitespace (`re.sub(r'\\s+', ' ', ...)`),
destroying newlines/indentation in anything multi-line, e.g. a code block, with no way for expand() to
restore it; (2) fragments were extracted per-pattern (FLUFF_PATTERNS list order) rather than in the
order they occur in the text, so two removed phrases could come back in the wrong order relative to
each other. Neither was caught before because the only prior test (`tests/test_integration.py`, kept
as-is) only checks substrings and length, never exact equality, on a single-sentence input with no
adjacent fluff phrases and no structured whitespace - exactly the two conditions that hid these bugs.
"""
import random

import pytest

from core.caveman import CavemanCompressor


def roundtrip(text: str) -> str:
    compressed, fragments = CavemanCompressor.compress(text)
    return CavemanCompressor.expand(compressed, fragments)


# ── the exact cases that used to fail ───────────────────────────────────────

def test_plain_fluffy_prose_round_trips_exactly():
    original = (
        "Certainly! I have analyzed the files and found that the bug is in line 42. "
        "I'm sorry for the inconvenience. Please let me know if you need further assistance."
    )
    compressed, fragments = CavemanCompressor.compress(original)
    assert compressed == "I have analyzed the files and found that the bug is in line 42."
    assert "Certainly!" not in compressed
    assert CavemanCompressor.expand(compressed, fragments) == original


def test_two_adjacent_closing_phrases_come_back_in_the_original_order():
    """The exact case that exposed the reordering bug: 'sorry' is written before 'please...', but
    the two patterns are checked in the other order in FLUFF_PATTERNS, and the two matches are
    textually adjacent (nothing between them), which used to make them swap."""
    original = "Done. I'm sorry for the inconvenience. Please let me know if you need further assistance."
    assert roundtrip(original) == original


def test_three_adjacent_closing_phrases_stay_in_order():
    original = "Done. I'm sorry for the inconvenience. I apologize for the inconvenience. Thank you."
    assert roundtrip(original) == original


def test_multiline_indented_text_preserves_every_character():
    """The exact case that exposed the whitespace-collapse bug: code has meaningful newlines and
    indentation that a blanket \\s+ -> ' ' collapse destroys and can never restore."""
    original = "Certainly! Here is the fix:\n\ndef foo():\n    if True:\n        return 1\n    return 0\n"
    compressed, _ = CavemanCompressor.compress(original)
    assert "\n" in compressed and "    " in compressed  # structure survives compression, not just expansion
    assert roundtrip(original) == original


def test_fenced_code_block_round_trips_exactly():
    original = "Of course! ```python\ndef f():\n    pass\n```\nThank you."
    assert roundtrip(original) == original


@pytest.mark.parametrize("original", [
    "no fluff here at all, just plain text",
    "  \n\t  leading and trailing whitespace preserved  \n\n",
    "",
    "Certainly! Of course! two prefixes in a row",
])
def test_other_edge_cases_round_trip_exactly(original):
    assert roundtrip(original) == original


# ── the observable behavior the README describes ────────────────────────────

def test_compress_removes_matched_phrases_and_shortens_the_text():
    original = "Certainly! The answer is 42. Thank you."
    compressed, fragments = CavemanCompressor.compress(original)
    assert compressed == "The answer is 42."
    assert len(compressed) < len(original)
    assert fragments  # something was tracked for reversal


def test_compress_is_a_no_op_when_nothing_matches():
    original = "The answer is 42."
    compressed, fragments = CavemanCompressor.compress(original)
    assert compressed == original and fragments == []


def test_expand_with_no_fragments_returns_the_text_unchanged():
    assert CavemanCompressor.expand("anything", []) == "anything"


def test_empty_input():
    assert CavemanCompressor.compress("") == ("", [])
    assert CavemanCompressor.expand("", []) == ""


# ── CavemanProtocol: only compresses for a non-user target, and only there ──

def test_protocol_only_compresses_for_a_non_user_target():
    from core.caveman import CavemanProtocol

    protocol = CavemanProtocol(enabled=True)
    original = "Certainly! The answer is 42. Thank you."
    assert protocol.process_outgoing(original, target="user") == original
    compressed = protocol.process_outgoing(original, target="agent")
    assert compressed == "The answer is 42."
    assert protocol.process_incoming(compressed) == original


def test_protocol_disabled_never_compresses():
    from core.caveman import CavemanProtocol

    protocol = CavemanProtocol(enabled=False)
    original = "Certainly! The answer is 42. Thank you."
    assert protocol.process_outgoing(original, target="agent") == original


# ── randomized property check: round trip holds for any combination ────────

def test_roundtrip_property_holds_across_many_random_combinations():
    fluff = [
        "Certainly! ", "Of course! ", "I'm sorry, ", "I apologize, ", "Here is the result: ",
        "Based on the provided context, ", " Thank you.", " I hope this helps.",
        " I'm sorry for the inconvenience.", " I apologize for the inconvenience.",
        " please let me know if you need further assistance.",
    ]
    body = [
        "def f():\n    return 1\n", "some prose here", "  indented\n\tmixed\ttabs  ",
        "line1\nline2\nline3", "", "x" * 5, "A sentence with punctuation! And another?",
    ]
    rng = random.Random(1234)  # fixed seed: deterministic, not flaky
    failures = []
    for _ in range(500):
        parts = [rng.choice(fluff if rng.random() < 0.5 else body) for _ in range(rng.randint(0, 6))]
        original = "".join(parts)
        result = roundtrip(original)
        if result != original:
            failures.append((original, result))
    assert not failures, f"{len(failures)}/500 round trips failed, e.g. {failures[0]!r}"
