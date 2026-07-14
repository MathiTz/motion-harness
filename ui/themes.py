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
        panel=d["border"],
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