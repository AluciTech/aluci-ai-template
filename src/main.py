from fire import Fire
from utils.config_utils import ConfigUtils
from configs.app_config import AppConfig
from utils.log_utils import setup_logs


def run(config_name: str = "settings"):
    ConfigUtils(config_name)
    AppConfig.load()
    setup_logs()


if __name__ == "__main__":
    Fire(run)
