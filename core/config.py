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
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value
        with open(self.config_path, 'w') as f:
            yaml.dump(self.data, f)

    def get_provider_config(self, provider_id: str) -> Dict[str, Any]:
        providers = self.data.get("providers", {})
        return providers.get(provider_id, {})
