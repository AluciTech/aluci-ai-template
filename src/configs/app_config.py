from dataclasses import dataclass
from pathlib import Path
from typing import Any
from utils.hydra_utils import HydraUtils


@dataclass
class AppConfig:
    settings: Any | None = None

    @classmethod
    def load(
        cls,
        config_dir: str | Path | None = None,
        config_file: str | None = None,
        overrides: list[str] | None = None,
    ):
        cls.settings = HydraUtils.load_config(config_dir, config_file, overrides)

    @classmethod
    def update(
        cls,
        updates: dict[str, Any],
        config_dir: str | Path | None = None,
        config_file: str | None = None,
    ):
        cls.settings = HydraUtils.update_config(updates, config_dir, config_file)
