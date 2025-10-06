from dataclasses import dataclass
from pathlib import Path
from typing import Any
from utils.config_utils import ConfigUtils


@dataclass
class AppConfig:
    settings: Any | None = None

    @classmethod
    def load(cls):
        cls.settings = ConfigUtils.load_config()

    @classmethod
    def update(
        cls,
        updates: dict[str, Any],
        config_dir: str | Path | None = None,
        config_name: str | None = None,
    ):
        cls.settings = ConfigUtils.update_config(updates, config_dir, config_name)
