"""Local API-key store (opencode-style auth).

Keys are stored in a per-user JSON file with ``0600`` permissions, keeping
secrets out of ``config.yml`` and the shell environment. Lookup order when
resolving a key for a provider: **auth store → environment variable →
config.yml**.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

AUTH_DIR = Path(os.getenv("MOTION_AUTH_DIR", str(Path.home() / ".config" / "motion-harness")))
AUTH_FILE = AUTH_DIR / "auth.json"


def _load() -> Dict[str, str]:
    if not AUTH_FILE.exists():
        return {}
    try:
        with open(AUTH_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: Dict[str, str]) -> None:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUTH_FILE, "w") as f:
        json.dump(data, f, indent=2)
    try:
        os.chmod(AUTH_FILE, 0o600)
    except Exception:
        pass


def set_key(provider: str, key: str) -> None:
    """Store (or overwrite) an API key for a provider."""
    data = _load()
    data[provider] = key
    _save(data)


def get_key(provider: str) -> Optional[str]:
    """Return the stored key for a provider, or None."""
    return _load().get(provider)


def remove_key(provider: str) -> bool:
    """Remove a stored key. Returns True if a key was removed."""
    data = _load()
    if provider in data:
        del data[provider]
        _save(data)
        return True
    return False


def list_keys() -> Dict[str, str]:
    """Return all stored keys as {provider: key}."""
    return _load()
