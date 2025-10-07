from typing import Any
import numpy as np
from tensorboard import program
import subprocess

from torch import Tensor

from utils.log_utils import log
from torch.utils.tensorboard import SummaryWriter


def launch_tensorboard(*, log_dir: str) -> None:
    """
    Launch TensorBoard with the specified log directory and attempt to open it in the default web browser.

    This method initializes a TensorBoard instance configured to use the provided log directory,
    launches TensorBoard, and logs the URL where it is accessible. It then attempts to open the URL
    in the user's default web browser using the "xdg-open" command. If opening the browser fails,
    an error message is logged instructing the user to manually visit the URL.

    Note:
        The automatic opening of the browser using "xdg-open" has only been verified to work on Ubuntu.
        It was not tested on other Linux distributions and does not work on MacOS or Windows.

    Parameters:
        log_dir (str): The path to the directory containing log files for TensorBoard.

    Returns:
        None
    """
    log_dir = str(log_dir)  # Ensure log_dir is a string

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir])
    url = tb.launch()
    log(message=f"TensorBoard started at {url}", level="INFO")
    try:
        subprocess.run(["xdg-open", url], check=True)
    except subprocess.CalledProcessError:
        log(
            message=f"Failed to open browser. Please manually visit: {url}",
            level="WARNING",
        )
