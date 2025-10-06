from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch import nn, Tensor

from configs.path_config import path_config
from decorators.error_handler import catch_errors
from schemas.model_schema import AbstractModel
from utils.log_utils import log
from hydra.utils import get_class, get_method


class MLP(nn.Module):
    """Multilayer Perceptron (MLP) model."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 10,
        activation: nn.Module | str = nn.ReLU(),
    ) -> None:
        super().__init__()

        if isinstance(activation, str):
            activation = get_class(activation)()  # Instantiate the activation function

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)


@dataclass
class MLPModel(AbstractModel):
    """A tiny multilayer perceptron where we only load the model non-trained."""

    def __post_init__(self) -> None:
        self.config_dir = self.config_dir or Path(path_config.src) / "models" / "mlp"
        self.config_name = self.config_name or "mlp_config.yaml"
        self.config = self._load_config()
        self.load_model()
        log(message=f"{self} initialized.", level="INFO")

    @catch_errors()
    def load_model(self) -> nn.Module:
        """Load the MLP model.

        Returns:
            nn.Module: The loaded MLP model.
        """
        log(message=f"Loading {self}...", level="INFO")
        self.model = MLP(**self.config["model_params"])
        log(message=f"{self} loaded and ready to use !", level="SUCCESS")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
