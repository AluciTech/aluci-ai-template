from dataclasses import dataclass, asdict
import os
from datetime import datetime
from utils.file_utils import get_project_root, create_dir

# Datetime
now = datetime.now()
run_date = now.strftime("%Y%m%d")
run_time = now.strftime("%H%M%S")


@dataclass
class PathSchema:
    root: str = get_project_root()
    data: str = os.path.join(root, "data")
    models: str = os.path.join(root, "models")

    # Logs
    logs: str = os.path.join(root, "logs")
    ## Log folder for current run
    run_log: str = os.path.join(logs, run_date, run_time)
    ## Config backup
    config_backup: str = os.path.join(run_log, "configs")

    # src
    src: str = os.path.join(root, "src")
    configs: str = os.path.join(src, "configs")

    def __post_init__(self):
        # Create missing folders
        [
            create_dir(dir_path=path, exist_ok=True)
            for var_name, path in asdict(self).items()
            if var_name != "root"
        ]
