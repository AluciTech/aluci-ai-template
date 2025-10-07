from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from configs.path_config import path_config
from decorators.error_handler import catch_errors
from schemas.model_schema import AbstractHFModel
from utils.hf_utils import get_huggingface_model
from utils.log_utils import log


@dataclass
class SmolLM3Model(AbstractHFModel):
    """A tiny HuggingFace SmolLM3 model."""

    def __post_init__(self) -> None:
        self.config_dir = (
            self.config_dir or Path(path_config.src) / "models" / "smol_lm3"
        )
        self.config_file = self.config_file or "smol_lm3_config.yaml"
        self.config = self._load_config()
        self.load_model()
        log(message=f"{self} initialized.", level="INFO")

    @catch_errors()
    def load_model(self) -> tuple[Any, Any]:
        """Load the SmolLM3 model."""

        self.tok_or_proc, self.model = get_huggingface_model(
            **self.config["model_params"]
        )
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # If user passed device_map in YAML, HF may already place weights on devices.
            # We still move the top module if single-device.
            if not hasattr(self.model, "hf_device_map"):
                self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
