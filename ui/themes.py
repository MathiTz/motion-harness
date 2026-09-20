"""Motion Harness theme definitions using Textual's native Theme system.

Each theme is registered with the Textual App via ``register_theme()``.
Setting ``app.theme = "one_dark"`` cascades through every CSS ``$variable``
in the entire widget tree — no manual style-patching needed.
"""

from __future__ import annotations

from typing import Dict, List

from textual.theme import Theme as TextualTheme


# ── Theme definitions ────────────────────────────────────────────────────────
# Each dict provides all the colours needed to build both:
#   1. A Textual-native Theme (for app.theme = "one_dark")
#   2. A lightweight dataclass-style Theme (backward compat for ThemeRegistry)

_RAW_THEMES: List[dict] = [
    dict(
        id="omni_dark",
        name="Omni Dark",
        background="#191622",
        surface="#23212B",
        panel="#2A2734",
        foreground="#E1E1E6",
        primary="#78D1E1",
        secondary="#483C67",
        accent="#78D1E1",
        border="#41414D",
        highlight="#2E2A3A",
        error="#E96379",
        success="#67E480",
        warning="#E89E64",
        dark=True,
    ),
    dict(
        id="one_dark",
        name="One Dark",
        background="#282c34",
        surface="#2c313a",
        foreground="#abb2bf",
        primary="#61afef",
        secondary="#5c6370",
        accent="#61afef",
        border="#3e4451",
        highlight="#3e4451",
        error="#e06c75",
        success="#98c379",
        warning="#e5c07b",
        dark=True,
    ),
    dict(
        id="solarized_light",
        name="Solarized Light",
        background="#fdf6e3",
        surface="#eee8d5",
        foreground="#657b83",
        primary="#268bd2",
        secondary="#93a1a1",
        accent="#268bd2",
        border="#d3cbb7",
        highlight="#eee8d5",
        error="#dc322f",
        success="#859900",
        warning="#b58900",
        dark=False,
    ),
    dict(
        id="dracula",
        name="Dracula",
        background="#282a36",
        surface="#44475a",
        foreground="#f8f8f2",
        primary="#bd93f9",
        secondary="#6272a4",
        accent="#bd93f9",
        border="#44475a",
        highlight="#44475a",
        error="#ff5555",
        success="#50fa7b",
        warning="#f1fa8c",
        dark=True,
    ),
    dict(
        id="nord",
        name="Nord",
        background="#2e3440",
        surface="#3b4252",
        foreground="#d8dee9",
        primary="#88c0d0",
        secondary="#4c566a",
        accent="#88c0d0",
        border="#3b4252",
        highlight="#434c5e",
        error="#bf616a",
        success="#a3be8c",
        warning="#ebcb8b",
        dark=True,
    ),
    # opencode native theme — matches the terminal UI from github.com/anomalyco/opencode
    dict(
        id="opencode",
        name="OpenCode",
        background="#0a0a0a",
        surface="#1e1e1e",
        panel="#141414",
        foreground="#eeeeee",
        primary="#fab283",
        secondary="#5c9cf5",
        accent="#9d7cd8",
        border="#484848",
        highlight="#282828",
        error="#e06c75",
        success="#7fd88f",
        warning="#f5a742",
        dark=True,
    ),
    # Catppuccin (catppuccin.com) — all four official flavors, using each
    # palette's own mauve/blue/red/green/peach for accent/secondary/error/
    # success/warning so the pastel identity carries through the cascade.
    dict(
        id="catppuccin_mocha",
        name="Catppuccin Mocha",
        background="#1e1e2e",
        surface="#313244",
        panel="#181825",
        foreground="#cdd6f4",
        primary="#cba6f7",
        secondary="#89b4fa",
        accent="#f5c2e7",
        border="#45475a",
        highlight="#585b70",
        error="#f38ba8",
        success="#a6e3a1",
        warning="#f9e2af",
        dark=True,
    ),
    dict(
        id="catppuccin_macchiato",
        name="Catppuccin Macchiato",
        background="#24273a",
        surface="#363a4f",
        panel="#1e2030",
        foreground="#cad3f5",
        primary="#c6a0f6",
        secondary="#8aadf4",
        accent="#f5bde6",
        border="#494d64",
        highlight="#5b6078",
        error="#ed8796",
        success="#a6da95",
        warning="#eed49f",
        dark=True,
    ),
    dict(
        id="catppuccin_frappe",
        name="Catppuccin Frapp\u00e9",
        background="#303446",
        surface="#414559",
        panel="#292c3c",
        foreground="#c6d0f5",
        primary="#ca9ee6",
        secondary="#8caaee",
        accent="#f4b8e4",
        border="#51576d",
        highlight="#626880",
        error="#e78284",
        success="#a6d189",
        warning="#e5c890",
        dark=True,
    ),
    dict(
        id="catppuccin_latte",
        name="Catppuccin Latte",
        background="#eff1f5",
        surface="#e6e9ef",
        panel="#dce0e8",
        foreground="#4c4f69",
        primary="#8839ef",
        secondary="#1e66f5",
        accent="#ea76cb",
        border="#ccd0da",
        highlight="#bcc0cc",
        error="#d20f39",
        success="#40a02b",
        warning="#df8e1d",
        dark=False,
    ),
]


# ── Build registries ─────────────────────────────────────────────────────────

class LightweightTheme:
    """Minimal theme object used by old code that references .background etc."""
    def __init__(self, name: str, background: str, foreground: str,
                 accent: str, secondary: str, border: str, highlight: str):
        self.name = name
        self.background = background
        self.foreground = foreground
        self.accent = accent
        self.secondary = secondary
        self.border = border
        self.highlight = highlight


_TEXTUAL_THEMES: Dict[str, TextualTheme] = {}
_LIGHT_THEMES: Dict[str, LightweightTheme] = {}

for d in _RAW_THEMES:
    tid = d["id"]

    # Textual-native theme
    _TEXTUAL_THEMES[tid] = TextualTheme(
        name=tid,
        primary=d["primary"],
        secondary=d["secondary"],
        background=d["background"],
        surface=d["surface"],
        foreground=d["foreground"],
        accent=d["accent"],
        error=d["error"],
        success=d["success"],
        warning=d["warning"],
        dark=d["dark"],
        panel=d.get("panel", d["border"]),
        boost=d["highlight"],
    )

    # Lightweight theme
    _LIGHT_THEMES[tid] = LightweightTheme(
        name=d["name"],
        background=d["background"],
        foreground=d["foreground"],
        accent=d["accent"],
        secondary=d["secondary"],
        border=d["border"],
        highlight=d["highlight"],
    )


class ThemeRegistry:
    """Backward-compatible registry. Also exposes Textual-native themes."""

    THEMES = _LIGHT_THEMES  # lightweight .background / .foreground etc

    @classmethod
    def get_theme(cls, theme_name: str) -> LightweightTheme:
        return cls.THEMES.get(theme_name, cls.THEMES["one_dark"])

    @classmethod
    def get_textual_theme(cls, theme_name: str) -> TextualTheme:
        return _TEXTUAL_THEMES.get(theme_name, _TEXTUAL_THEMES["one_dark"])

    @classmethod
    def theme_ids(cls) -> List[str]:
        return list(_TEXTUAL_THEMES.keys())