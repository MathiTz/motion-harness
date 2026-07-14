import yaml
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class AppConfig:
    workspace_path: str
    default_theme: str
    max_parallel_tasks: Optional[int]
    providers: Dict[str, Any]
    default_provider: str

class ConfigManager:
    CONFIG_PATHS = ["config.yml", "config.example.yml"]

    def __init__(self, config_path: str = ""):
        if config_path:
            self.config_path = config_path
        else:
            # Try config.yml first, fall back to config.example.yml
            self.config_path = next((p for p in self.CONFIG_PATHS if os.path.exists(p)), self.CONFIG_PATHS[0])
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _env(self, key: str, default: Any = None) -> Any:
        """Resolve a value from environment variables first, then config."""
        return os.environ.get(key) or default

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value
        with open(self.config_path, 'w') as f:
            yaml.dump(self.data, f)

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

        # Resolve api_key from environment
        env_key = f"{base_id.replace('-', '_').upper()}_API_KEY"
        env_val = os.environ.get(env_key)
        if not env_val:
            prefix = base_id.split('-')[0].upper()
            generic_key = f"{prefix}_API_KEY"
            env_val = os.environ.get(generic_key)
        if env_val:
            config = {**config, "api_key": env_val}

        # Resolve model: explicit model_name > default_model > options.model
        models = config.get("models", {})
        if models:
            # Provider uses models list
            chosen_model = model_name or config.get("default_model") or next(iter(models))
            if chosen_model not in models:
                raise ValueError(f"Unknown model '{chosen_model}' for provider '{base_id}'. Available: {', '.join(models.keys())}")
            model_opts = models[chosen_model]
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

    def get_default_provider(self) -> str:
        return os.environ.get("MOTION_DEFAULT_PROVIDER") or self.data.get("providers", {}).get("default", "ollama-cloud")

    def has_api_key(self, provider_id: str) -> bool:
        """Check whether a provider has a usable API key (env var or config)."""
        if '/' in provider_id:
            base_id = provider_id.split('/', 1)[0]
        else:
            base_id = provider_id

        providers = self.data.get("providers", {})
        cfg = providers.get(base_id, {})

        # Check config file first
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

        return False

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
