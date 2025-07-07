import random
import numpy as np
import torch
import pytest

from utils.seed_utils import set_seed


@pytest.mark.parametrize("seed", [0, 1337, 42])
def test_set_seed_is_deterministic(seed):
    """
    Running set_seed twice with the *same* seed must reproduce:
      - random.random()
      - np.random.rand()
      - torch.rand() (CPU)
      - torch.rand() (CUDA)
    """
    # First run
    set_seed(seed)
    py_1 = random.random()
    np_1 = np.random.rand()
    t_cpu1 = torch.rand(4)  # 1-D tensor of 4 numbers
    t_gpu1 = torch.rand(4, device="cuda") if torch.cuda.is_available() else None

    # Second run (same seed)
    set_seed(seed)
    assert random.random() == py_1
    assert np.random.rand() == np_1
    assert torch.equal(torch.rand(4), t_cpu1)

    if torch.cuda.is_available():
        assert torch.equal(torch.rand(4, device="cuda"), t_gpu1)


def test_different_seeds_produce_different_sequences():
    """
    Two distinct seeds should *not* give identical sequences
    (extremely unlikely collisions ignored).
    """
    set_seed(123)
    py_a = random.random()
    np_a = np.random.rand()
    t_cpu_a = torch.rand(4)

    set_seed(456)
    assert random.random() != py_a
    assert np.random.rand() != np_a
    assert not torch.equal(torch.rand(4), t_cpu_a)
