from squadro.tools.constants import inf


def get_expected_score(a, b):
    """
    Expected score of A vs. B.

    >>> round(get_expected_score(30, 10), 4)
    0.5288

    >>> get_expected_score(30, 10) + get_expected_score(10, 30)
    1.0

    >>> get_expected_score(10, 10)
    0.5

    >>> get_expected_score(inf, 0)
    1.0

    >>> get_expected_score(0, inf)
    0.0
    """
    return 1 / (1 + 10 ** ((b - a) / 400))
