import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import squadro


@dataclass
class DefaultParams:
    agent = "random"
    time_out = 900.0
    first = -1
    n_pawns = 5
    max_time_per_move = .005
    uct = 1.0
    mcts_method = 'biased_uct'

    @classmethod
    def get_uct(cls, n_pawns=None):
        # return cls.uct.get(n_pawns, 1.0)
        return cls.uct

    def __str__(self):
        return str(self.__dict__)

    @classmethod
    def attributes(cls):
        return {
            k: getattr(cls, k)
            for k in dir(cls)
            if not k.startswith('_') and not callable(getattr(cls, k))
        }

    @classmethod
    def print(cls):
        print(cls.attributes())

    @classmethod
    @contextmanager
    def update(cls, **kwargs):
        old = cls.attributes()
        for key, value in kwargs.items():
            setattr(cls, key, value)
        try:
            yield
        finally:
            for key, value in old.items():
                setattr(cls, key, value)

RESOURCE_PATH = Path(squadro.__file__).parent / 'resources'
MAX_INT = sys.maxsize
inf = float("inf")
