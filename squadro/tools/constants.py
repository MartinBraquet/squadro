from pathlib import Path

import squadro


class DefaultParams:
    agent = "random"
    time_out = 900.0
    first = -1
    n_pawns = 5


RESOURCE_PATH = Path(squadro.__file__).parent / 'resources'
