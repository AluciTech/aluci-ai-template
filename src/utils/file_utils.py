import subprocess
import os
import shutil
import threading
from datetime import datetime
from typing import Any, List, Literal
from utils.log_utils import log
from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

# A set of common image file extensions for quick lookup.
IMG_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}

_lock = threading.Lock()

# A special YAML instance from ruamel.yaml to preserve comments and formatting.
yaml = YAML()
yaml.default_flow_style = False
yaml.preserve_quotes = True


def get_project_root() -> str:
    """Finds the root directory of the current Git repository.

    This function executes a Git command to determine the top-level directory
    of the project.

    Returns:
        str: The absolute path to the project's root directory.

    Raises:
        RuntimeError: If the current directory is not a Git repository or if
                      Git is not installed.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=".",
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Not a git repository or git is not installed.") from e


def rename_dir(*, dir_path: Path | str, new_dir_path: Path | str) -> str:
    """Renames a directory in a thread-safe manner.

    Args:
        dir_path (Path | str): The current path of the directory.
        new_dir_path (Path | str): The new path for the directory.

    Returns:
        str: The new path of the directory upon successful rename.

    Raises:
        RuntimeError: If the renaming operation fails for any reason.
    """
    with _lock:
        try:
            os.rename(dir_path, new_dir_path)
            log(
                message=f"Renamed directory: {dir_path} -> {new_dir_path}",
                level="SUCCESS",
            )
            return str(new_dir_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to rename directory {dir_path} to {new_dir_path}: {e}"
            ) from e


def rename_dir_if_exists(*, dir_path: Path | str) -> str:
    """Renames a directory by appending a timestamp if it already exists.

    If the directory at `dir_path` exists, it will be renamed to
    `{dir_path}_{YYYYMMDD_HHMMSS}`.

    Args:
        dir_path (Path | str): The path of the directory to check and rename.

    Returns:
        str: The new, timestamped directory path if renamed, otherwise the
             original `dir_path`.
    """
    if os.path.exists(dir_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_path = f"{dir_path}_{timestamp}"
        return rename_dir(dir_path=dir_path, new_dir_path=new_dir_path)
    else:
        log(message=f"Directory does not exist: {dir_path}", level="WARNING")
        return str(dir_path)


def create_dir(*, dir_path: Path | str, exist_ok: bool = True) -> None:
    """Creates a new directory in a thread-safe manner.

    This function can create intermediate directories as needed.

    Args:
        dir_path (Path | str): The path of the directory to be created.
        exist_ok (bool, optional): If False, a FileExistsError is raised if the
                                   target directory already exists. Defaults to True.

    Raises:
        RuntimeError: If the directory creation fails.
    """
    with _lock:
        try:
            path_exists = os.path.exists(dir_path)
            os.makedirs(dir_path, exist_ok=exist_ok)
            if not path_exists:
                log(message=f"Created directory: {dir_path}", level="SUCCESS")
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {dir_path}: {e}") from e


def remove_dir(*, dir_path: Path | str) -> None:
    """Removes a directory and all its contents recursively and thread-safely.

    Args:
        dir_path (Path | str): The path of the directory to be removed.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        RuntimeError: If the directory removal fails for any other reason.
    """
    with _lock:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                log(message=f"Removed directory: {dir_path}", level="SUCCESS")
            except Exception as e:
                raise RuntimeError(f"Failed to remove directory {dir_path}: {e}") from e
        else:
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")


def rename_file_if_exists(*, file_path: Path | str) -> str:
    """Generates a new, non-existent filename if the given path already exists.

    If `file.txt` exists, this function will try `file_1.txt`, `file_2.txt`,
    and so on, until it finds a name that is not in use.

    Args:
        file_path (Path | str): The original file path to check.

    Returns:
        str: A unique file path. Returns the original path if it doesn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        return str(path)

    base = path.stem
    suff = path.suffix
    n = 1
    while True:
        new_path = path.with_name(f"{base}_{n}{suff}")
        if not new_path.exists():
            return str(new_path)
        n += 1


def copy_file(*, src: Path | str, dest: Path | str) -> None:
    """Copies a file from a source to a destination in a thread-safe manner.

    Args:
        src (Path | str): The path of the source file.
        dest (Path | str): The path of the destination. This can be a directory
                           or a new file name.

    Raises:
        FileNotFoundError: If the source file does not exist.
        RuntimeError: If the copy operation fails.
    """
    with _lock:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
        try:
            shutil.copy(src, dest)
            log(message=f"Copied file from {src} to {dest}", level="SUCCESS")
        except Exception as e:
            raise RuntimeError(f"Failed to copy file from {src} to {dest}: {e}") from e


def get_file_extension(path: Path | str) -> str:
    """Extracts the file extension from a path.

    Args:
        path (Path | str): The file path.

    Returns:
        str: The file extension without the leading dot (e.g., "txt", "yaml").
    """
    return str(Path(path).suffix).removeprefix(".")


def _read_yaml(path: str | Path) -> dict:
    """Helper function to read and parse a YAML file.

    Args:
        path (str | Path): The path to the YAML file.

    Returns:
        dict: The data loaded from the YAML file.
    """
    with open(path, "r") as yaml_file:
        data = yaml.load(yaml_file)
    return data


def read_file(*, path: Path | str) -> Any:
    """Reads data from a file, dispatching to the appropriate parser based on extension.

    Currently only supports YAML files.

    Args:
        path (Path | str): The path to the file to be read.

    Returns:
        Any: The content of the file (e.g., a dict for YAML).

    Raises:
        NotImplementedError: If the file extension is not supported.
    """
    match get_file_extension(path):
        case "yaml":
            data = _read_yaml(path)
            log(message=f"Data were read successfully from '{path}'", level="SUCCESS")
            return data
        case _:
            log(message="Extension not implemented yet", level="ERROR")
            raise NotImplementedError


def _write_yaml(path: str | Path, new_data: dict[str, Any]) -> None:
    """Helper function to update and write data to a YAML file thread-safely.

    This function reads the existing file, updates its values with `new_data`,
    and writes it back, preserving formatting. Dot-separated keys in `new_data`
    (e.g., "parent.child") are used to update nested values.

    Args:
        path (str | Path): The path to the YAML file.
        new_data (dict[str, Any]): A dictionary containing the data to update.
    """
    with _lock:
        data = read_file(path=path)

        for key, value in new_data.items():
            if hasattr(value, "tolist"):
                value = value.tolist()  # Handle numpy/pandas data types

            if (
                isinstance(value, list)
                and value
                and all(isinstance(i, list) for i in value)
            ):
                styled_list = CommentedSeq()

                for sub_list in value:
                    flow_seq = CommentedSeq(sub_list)
                    flow_seq.fa.set_flow_style()
                    styled_list.append(flow_seq)

                value = styled_list

            keys = key.split(".")
            current_level = data
            for k in keys[:-1]:
                current_level = current_level[k]
            current_level[keys[-1]] = value

        with open(path, "w") as yaml_file:
            yaml.dump(data, yaml_file)


def write_file(*, path: Path | str, data: Any) -> None:
    """Writes data to a file, dispatching to the appropriate writer based on extension.

    Currently only supports updating YAML files.

    Args:
        path (Path | str): The path of the file to write to.
        data (Any): The data to be written.

    Raises:
        NotImplementedError: If the file extension is not supported.
    """
    match get_file_extension(path):
        case "yaml":
            _write_yaml(path, data)
            log(message=f"Data successfully written in '{path}'", level="SUCCESS")
        case _:
            log(message="Extension not implemented yet", level="ERROR")
            raise NotImplementedError


def _list_images(path: str | Path) -> List[str]:
    """Helper function to list all image files in a directory.

    Args:
        path (str | Path): The path to the directory.

    Returns:
        List[str]: A list of filenames that have a recognized image extension.
    """
    return [f for f in os.listdir(path) if Path(f).suffix.lower() in IMG_EXT]


def list_files(
    *, path: str | Path, file_type: Literal["images"] | None = None
) -> List[str]:
    """Lists files in a given directory, with an option to filter by type.

    Args:
        path (str | Path): The path to the directory.
        file_type (Literal["images"] | None, optional): The type of files to list.
            If "images", only image files are returned. If None, all files and
            directories are returned. Defaults to None.

    Returns:
        List[str]: A list of filenames found in the directory.
    """
    match file_type:
        case "images":
            return _list_images(path)
        case _:
            return os.listdir(path)
