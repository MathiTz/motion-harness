from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Theme:
    name: str
    background: str
    foreground: str
    accent: str
    secondary: str
    border: str
    highlight: str

class ThemeRegistry:
    # Colors inspired by popular VS Code themes
    THEMES = {
        "one_dark": Theme(
            name="One Dark",
            background="#282c34",
            foreground="#abb2bf",
            accent="#61afef",
            secondary="#5c6370",
            border="#3e4451",
            highlight="#3e4451"
        ),
        "solarized_light": Theme(
            name="Solarized Light",
            background="#fdf6e3",
            foreground="#657b83",
            accent="#268bd2",
            secondary="#93a1a1",
            border="#eee8d5",
            highlight="#eee8d5"
        ),
        "dracula": Theme(
            name="Dracula",
            background="#282a36",
            foreground="#f8f8f2",
            accent="#bd93f9",
            secondary="#6272a4",
            border="#44475a",
            highlight="#44475a"
        ),
        "nord": Theme(
            name="Nord",
            background="#2e3440",
            foreground="#d8dee9",
            accent="#88c0d0",
            secondary="#4c566a",
            border="#3b4252",
            highlight="#434c5e"
        )
    }

    @classmethod
    def get_theme(cls, theme_name: str) -> Theme:
        return cls.THEMES.get(theme_name, cls.THEMES["one_dark"])
