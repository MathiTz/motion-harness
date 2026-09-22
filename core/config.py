import yaml
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass

from core.catalog import merge_catalog
from core import auth

@dataclass
class AppConfig:
    workspace_path: str
    default_theme: str
    max_parallel_tasks: Optional[int]
    providers: Dict[str, Any]
    default_provider: str

# The installed location of the harness (parent of core/), not the caller's
# CWD. `motion` can be pointed at any workspace directory, so config lookup
# must not depend on where it happens to be invoked from.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConfigManager:
    CONFIG_PATHS = [
        os.path.join(_REPO_DIR, "config.yml"),
        os.path.join(_REPO_DIR, "config.example.yml"),
    ]

    def __init__(self, config_path: str = ""):
        if config_path:
            self.config_path = config_path
        elif os.environ.get("MOTION_CONFIG"):
            # Explicit config file (used by scripts/CI and the end-to-end tests).
            self.config_path = os.environ["MOTION_CONFIG"]
        else:
            # Try config.yml first, fall back to config.example.yml
            self.config_path = next((p for p in self.CONFIG_PATHS if os.path.exists(p)), self.CONFIG_PATHS[0])
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        # Merge the built-in catalog so pre-configured providers/models are
        # always available, with user config taking precedence.
        providers = data.get("providers", {})
        merged_providers = merge_catalog(providers)
        data["providers"] = merged_providers
        return data

    def reload(self) -> None:
        """Reload config from disk and re-merge with the current catalog."""
        self.data = self._load_config()

    def _env(self, key: str, default: Any = None) -> Any:
        """Resolve a value from environment variables first, then config."""
        return os.environ.get(key) or default

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Persist a single top-level key (e.g. ``default_theme``,
        ``last_provider``) to config.yml.

        Reads/writes the raw file directly rather than dumping ``self.data``
        wholesale - ``self.data["providers"]`` has the full built-in catalog
        merged in (see ``_load_config``), so writing it back would silently
        bloat config.yml with every built-in provider/model on the very
        first settings change (e.g. switching themes).
        """
        try:
            with open(self.config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raw = {}
        raw[key] = value
        with open(self.config_path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
        self.data[key] = value

    def get_provider_config(self, provider_id: str) -> Dict[str, Any]:
        """Resolve a provider config, supporting provider/model syntax.

        Examples:
          'ollama-cloud'           -> default model from default_model or options.model
          'ollama-cloud/gemma4:31b' -> ollama-cloud with model overridden to gemma4:31b
        """
        # Split provider/model if present
        if '/' in provider_id:
            base_id, model_name = provider_id.split('/', 1)
        else:
            base_id, model_name = provider_id, None

        providers = self.data.get("providers", {})
        config = providers.get(base_id, {})
        if not config:
            raise ValueError(f"Unknown provider: {base_id}")

        # Resolve api_key: auth store → env var → config.yml
        env_key = f"{base_id.replace('-', '_').upper()}_API_KEY"
        env_val = os.environ.get(env_key)
        if not env_val:
            prefix = base_id.split('-')[0].upper()
            generic_key = f"{prefix}_API_KEY"
            env_val = os.environ.get(generic_key)
        stored = auth.get_key(base_id)
        if stored:
            config = {**config, "api_key": stored}
        elif env_val:
            config = {**config, "api_key": env_val}

        # Resolve model: explicit model_name > default_model > options.model
        models = config.get("models", {})
        if models:
            # Provider uses models list. For cloud providers, allow arbitrary
            # model names (live providers add models faster than any hardcoded
            # catalog). For local providers, keep validation so typos surface.
            chosen_model = model_name or config.get("default_model") or next(iter(models))
            is_cloud = config.get("provider_type") == "cloud"
            if chosen_model not in models and not is_cloud:
                raise ValueError(f"Unknown model '{chosen_model}' for provider '{base_id}'. Available: {', '.join(models.keys())}")
            model_opts = models.get(chosen_model, {"temperature": 0.7, "max_tokens": 4096})
            config = {
                **config,
                "name": f"{config.get('name', base_id)} ({chosen_model})",
                "options": {"model": chosen_model, **model_opts},
            }
        elif model_name:
            # No models list, but user specified a model override
            config = {
                **config,
                "options": {**config.get("options", {}), "model": model_name},
            }

        return config

    def add_model(self, provider_id: str, model_name: str, options: Optional[Dict[str, Any]] = None) -> None:
        """Add a model entry to a provider and persist it to config.yml.

        Reads/writes the raw file directly (not self.data, which has the
        built-in catalog merged in) so this doesn't bloat config.yml with
        every catalog default the moment a single custom model is added.
        """
        model_name = (model_name or "").strip()
        if not model_name:
            raise ValueError("model name must not be empty")
        if not provider_id:
            raise ValueError("provider id must not be empty")

        try:
            with open(self.config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raw = {}
        raw_providers = raw.get("providers", {})
        provider_cfg = raw_providers.get(provider_id)
        if provider_cfg is None:
            # The provider may only exist via the built-in catalog merge (not
            # yet in the raw file) - confirm it's real, then persist a
            # minimal override that survives independently of the catalog.
            if provider_id not in self.data.get("providers", {}):
                raise ValueError(f"Unknown provider: {provider_id}")
            provider_cfg = {}
            raw_providers[provider_id] = provider_cfg
        models = provider_cfg.setdefault("models", {})
        if model_name not in models:
            models[model_name] = options or {"temperature": 0.7, "max_tokens": 4096}
        raw["providers"] = raw_providers
        with open(self.config_path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
        self.reload()

    def get_default_provider(self) -> str:
        """Resolve which provider/model to launch with, in priority order:
        an explicit env override, the last model the user switched to in
        the TUI (persisted via ``last_provider``), then the catalog's
        configured default.
        """
        return (
            os.environ.get("MOTION_DEFAULT_PROVIDER")
            or self.data.get("last_provider")
            or self.data.get("providers", {}).get("default", "ollama-cloud")
        )

    def has_api_key(self, provider_id: str) -> bool:
        """Check whether a provider has a usable API key (env var or config)."""
        if '/' in provider_id:
            base_id = provider_id.split('/', 1)[0]
        else:
            base_id = provider_id

        providers = self.data.get("providers", {})
        cfg = providers.get(base_id, {})

        # Check auth store first
        if auth.get_key(base_id):
            return True

        # Check config file
        config_key = cfg.get("api_key")
        if config_key:
            return True

        # Check environment variables
        env_key = f"{base_id.replace('-', '_').upper()}_API_KEY"
        if os.environ.get(env_key):
            return True
        prefix = base_id.split('-')[0].upper()
        generic_key = f"{prefix}_API_KEY"
        if os.environ.get(generic_key):
            return True

        # Local providers don't need a key
        if cfg.get("provider_type") == "local":
            return True

        # CLI delegates (Claude Code / Codex via the login already on this machine) don't take a key
        # either - "usable" means the binary is on PATH. See core/cli_delegate.py.
        if cfg.get("provider_type") == "cli":
            from core.cli_delegate import DELEGATE_CATALOG

            models = cfg.get("models") or {}
            delegate = next(iter(models.values()), {}).get("delegate", "") if models else ""
            checker = DELEGATE_CATALOG.get(delegate)
            return bool(checker and checker())

        return False

    def unavailable_reason(self, provider_id: str) -> str:
        """Empty string if the provider is usable; otherwise a short, specific reason plus how to fix
        it - shown in the model picker instead of the provider just silently not appearing there."""
        if self.has_api_key(provider_id):
            return ""
        base_id = provider_id.split('/', 1)[0]
        cfg = (self.data.get("providers", {})).get(base_id, {})
        if cfg.get("provider_type") == "cli":
            models = cfg.get("models") or {}
            delegate = next(iter(models.values()), {}).get("delegate", "") if models else ""
            binary = {"claude-cli": "claude", "codex-cli": "codex"}.get(delegate, delegate)
            return f"'{binary}' isn't on PATH - install it and run `{binary} login`"
        env_key = f"{base_id.replace('-', '_').upper()}_API_KEY"
        reason = f"no API key - `motion auth login {base_id}`, set {env_key}, or add it in Ctrl+A"
        if base_id in ("claude", "openai") and self.has_api_key("claude-cli" if base_id == "claude" else "codex-cli"):
            cli = "claude-cli" if base_id == "claude" else "codex-cli"
            reason += f"; or switch to {(self.data.get('providers', {}).get(cli) or {}).get('name', cli)}, which is already available"
        return reason

    def list_providers(self) -> list:
        """Return list of (provider_id, name, models, has_key) tuples."""
        providers = self.data.get("providers", {})
        default = self.get_default_provider()
        result = []
        for pid, cfg in providers.items():
            if pid == "default":
                continue
            models = list(cfg.get("models", {}).keys()) if "models" in cfg else [cfg.get("options", {}).get("model", "?")]
            has_key = self.has_api_key(pid)
            result.append((pid, cfg.get("name", pid), models, pid == default.split('/')[0], has_key))
        return result
