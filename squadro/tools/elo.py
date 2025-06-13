from squadro.tools.constants import inf
from squadro.tools.logs import training_logger as logger


class Elo:
    def __init__(self, start=0, step=0):
        self.current = start
        self.checkpoint = start
        self.history = {step: self.current}
        self.k = 2

    def __repr__(self):
        return f"{self.current:.0f} vs {self.checkpoint:.0f}"

    def update(self, win_rate, n, step):
        expected_score = get_expected_score(self.current, self.checkpoint)
        delta_elo = self.k * (win_rate - expected_score) * n
        self.delta_update(delta_elo, step)
        logger.info(f"Elo: {self} (delta: {delta_elo:.0f})")

    def delta_update(self, delta, step):
        self.current += delta
        self.checkpoint -= delta
        self.history[step] = self.current

    def update_checkpoint(self):
        self.checkpoint = self.current


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
