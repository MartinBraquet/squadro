import random

import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_random_index(probs: np.ndarray):
    """
    Get a random index from a probability distribution

    >>> get_random_index(np.array([1, 0]))
    np.int64(0)

    >>> get_random_index(np.array([0, 1]))
    np.int64(1)

    >>> np.random.seed(0)
    >>> get_random_index(np.array([.01] * 100))
    np.int64(52)
    """
    probs = probs.astype(np.float64)
    probs /= np.sum(probs)
    samples = np.random.multinomial(1, probs)
    index = np.where(samples == 1)[0][0]
    return index
