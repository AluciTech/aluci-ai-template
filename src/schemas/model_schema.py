from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from decorators.error_handler import catch_errors
from utils.hydra_utils import HydraUtils


@dataclass
class AbstractModel(ABC):
    """Abstract class for all model types."""

    config_dir: str | Path | None = None
    config_file: str | None = None
    config: dict[str, Any] = field(init=False)

    model: Any = field(init=False)

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    def _load_config(self) -> dict[str, Any]:
        """
        Returns the full config dict.
        Expected structure includes key 'model_params'.
        """
        return HydraUtils.load_config(
            config_dir=self.config_dir, config_file=self.config_file
        )


@dataclass
class AbstractHFModel(AbstractModel):
    """Abstract class for HuggingFace models."""

    tok_or_proc: Any = field(init=False)
    device: torch.device = field(init=False)

    def _create_prompt(self) -> None: ...
    def _infer(self) -> None: ...
    def generate(self) -> None: ...
