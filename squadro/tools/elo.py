from squadro.tools.constants import inf


class Elo:
    def __init__(self, start=0):
        self.current = start
        self.checkpoint = start

    def __repr__(self):
        return f"{self.current:.0f} vs {self.checkpoint:.0f}"

    def update(self, delta):
        self.current += delta
        self.checkpoint -= delta


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
