from dataclasses import dataclass
from pathlib import Path
from typing import Any
from hydra.utils import instantiate
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import shutil

from configs.path_config import path_config
from utils.file_utils import write_file


@dataclass
class HydraUtils:
    """
    A utility class for loading and managing configurations using Hydra and OmegaConf.

    Provides methods to:
    - Load a configuration from a YAML file with initial overrides.
    - Update the YAML file on disk and reload the configuration to make changes persistent.
    """

    config_dir: str | Path = path_config.configs
    config_file: str = "settings.yaml"

    _resolver_registered: bool = False

    @classmethod
    def _ensure_resolver(cls) -> None:
        """Register ${path_config:<attr>} -> getattr(path_config, <attr>) exactly once."""
        if not cls._resolver_registered:
            OmegaConf.register_new_resolver(
                "path_config",
                lambda key: getattr(path_config, key),
                use_cache=True,
            )
            cls._resolver_registered = True

    @classmethod
    def load_config(
        cls,
        config_dir: str | Path | None = None,
        config_file: str | None = None,
        overrides: list[str] | None = None,
    ) -> Any:
        """
        Loads and instantiates a configuration.

        Args:
            config_dir: The directory where the configuration file is located. Defaults to None.
            config_file: The name of the configuration file. Defaults to None.
            overrides: A list of strings to override specific configuration fields. Defaults to None.

        Returns:
            An instantiated Python object based on the configuration.

        Example:
            config = HydraUtils.load_config(
                config_dir="configs",
                config_file="settings.yaml",
                overrides=["++logs.level=DEBUG", "++model._target_=models.mlp.mlp_model.MLPModel"]
            )
        """
        cls._ensure_resolver()

        config_dir = config_dir or cls.config_dir
        config_file = config_file or cls.config_file
        config_path = Path(config_dir) / config_file

        # Make a backup of the config for future analysis
        shutil.copy2(config_path, Path(path_config.config_backup))

        # Handling cases where hydra is already initialized, leading to the following error:
        # `ValueError: GlobalHydra is already initialized, call GlobalHydra.instance().clear() if you want to re-initialize
        # Returning only the conf as a `dict` without recursive instantiation (`instantiate` method)
        if GlobalHydra.instance().is_initialized():
            return OmegaConf.load(config_path)

        with initialize_config_dir(str(config_dir), version_base=None):
            config = compose(config_file, overrides=overrides)

        return instantiate(config, _convert_="object")

    @classmethod
    def update_config(
        cls,
        updates: dict[str, Any],
        config_dir: str | Path | None = None,
        config_file: str | None = None,
    ) -> Any:
        """
        Updates configuration file on disk and instantiate the updated configuration.

        Args:
            updates (dict[str, Any]): Dictionary of fields to update.
            config_dir (str | Path | None, optional): The directory where the configuration file is located. Defaults to None.
            config_file (str | None, optional): The name of the configuration file. Defaults to None.

        Returns:
            Any: The instantiated updated configuration.
        """
        cls._ensure_resolver()

        config_dir = config_dir or cls.config_dir
        config_file = config_file or cls.config_file
        config_path = Path(config_dir) / config_file

        write_file(path=config_path, data=updates)

        return cls.load_config(config_dir, config_file)
