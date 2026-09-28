"""core/redact.py: best-effort stripping of secret-shaped values (issue #19). Pattern-based, not a
general DLP system - see the module's own docstring for what it deliberately does not catch."""
from core.redact import redact_text, redact_value


def test_key_value_pairs_with_a_secret_shaped_name_are_redacted():
    assert redact_text("ANTHROPIC_API_KEY=sk-ant-abcdefghijklmnopqrstuvwxyz1234567890") == "ANTHROPIC_API_KEY=[REDACTED]"
    assert redact_text('OPENAI_API_KEY="sk-abcdefghijklmnopqrstuvwx"') == 'OPENAI_API_KEY="[REDACTED]"'
    assert redact_text("DB_PASSWORD: hunter2superlongpassword") == "DB_PASSWORD: [REDACTED]"
    assert redact_text("my_github_token = ghp_abcdefghijklmnopqrstuvwxyz123456") == "my_github_token = [REDACTED]"


def test_bearer_tokens_are_redacted_but_the_scheme_stays_visible():
    out = redact_text("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def")
    assert out == "Authorization: Bearer [REDACTED]"


def test_bare_vendor_keys_are_redacted_even_without_a_name_label():
    assert "sk-ant-zzzzzzzzzzzzzzzzzzzzzzzzz" not in redact_text("random text with sk-ant-zzzzzzzzzzzzzzzzzzzzzzzzz embedded")
    assert "AKIAABCDEFGHIJKLMNOP" not in redact_text("AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP")


def test_plain_prose_mentioning_keys_is_left_alone():
    text = "nothing secret here, just plain prose about api keys in general"
    assert redact_text(text) == text


def test_empty_and_none_like_input_is_handled():
    assert redact_text("") == ""


def test_redact_value_walks_nested_dicts_and_lists():
    value = {
        "role": "tool",
        "content": [{"type": "text", "text": "TOKEN=abc123verysecretvalue"}],
        "meta": ["fine", "STRIPE_SECRET_KEY=zzzsupersecretvalue"],
    }
    out = redact_value(value)
    assert out["content"][0]["text"] == "TOKEN=[REDACTED]"
    assert out["meta"] == ["fine", "STRIPE_SECRET_KEY=[REDACTED]"]
    assert out["role"] == "tool"  # untouched, no secret shape


def test_redact_value_passes_through_non_string_types_unchanged():
    assert redact_value(42) == 42
    assert redact_value(None) is None
    assert redact_value(True) is True
    assert redact_value([1, 2, {"n": 3}]) == [1, 2, {"n": 3}]
