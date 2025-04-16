import numpy as np


def get_random_sample(probs: np.ndarray):
    """
    Get a random sample from a probability distribution

    >>> get_random_sample(np.array([1, 0]))
    np.int64(0)

    >>> get_random_sample(np.array([0, 1]))
    np.int64(1)

    >>> np.random.seed(0)
    >>> get_random_sample(np.array([.01] * 100))
    np.int64(52)
    """
    samples = np.random.multinomial(1, probs)
    index = np.where(samples == 1)[0][0]
    return index
