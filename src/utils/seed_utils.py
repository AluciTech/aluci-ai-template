import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Seed `random`, `numpy`, and `torch` (CPU & CUDA) for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Any integer seed you want to use.

    Example
    -------
    >>> from utils.seed_utils as seed_utils
    >>> set_seed(1234)
    """
    # Python’s random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (all CUDA GPUs)
    torch.cuda.manual_seed_all(seed)

    # Extra: make CuDNN deterministic (trades a bit of speed for bit-for-bit reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
