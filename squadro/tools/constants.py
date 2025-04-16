import sys
from pathlib import Path

import squadro


class DefaultParams:
    agent = "random"
    time_out = 900.0
    first = -1
    n_pawns = 5
    max_time_per_move = .005


RESOURCE_PATH = Path(squadro.__file__).parent / 'resources'
MAX_INT = sys.maxsize
inf = float("inf")
