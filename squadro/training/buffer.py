import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
from matplotlib import pyplot as plt

from squadro.state import State
from squadro.tools.basic import dict_factory
from squadro.tools.disk import load_pickle, dump_pickle
from squadro.tools.log import training_logger as logger
from squadro.training.constants import DQL_PATH


class ReplayBuffer:
    """
    Replay buffer stored in disk
    """

    def __init__(
        self,
        path: Path = None,
        n_pawns: int = None,
        max_size: int = None,
    ):
        if path is None:
            assert n_pawns is not None, "Must provide either path or n_pawns"
            path = DQL_PATH / f"replay_buffer_{n_pawns}.pkl"
        self.path = Path(path)
        self._results = None
        self.clear()
        self.load()
        self.max_size = int(max_size or 20e3)

    def __repr__(self):
        return f"{len(self)} samples @ {self.path}"

    def __len__(self):
        return sum(self.lengths)

    @property
    def data(self):
        return self._results['data']

    @property
    def diversity_history(self):
        return self._results['diversity_history']

    @property
    def lengths(self):
        return [len(data) for w, f, data in self.iter_buckets()]

    @property
    def sample_ratio(self):
        return {f"{w=}, {f=}": len(data) / len(self) for w, f, data in self.iter_buckets()}

    def append(self, sample, winner: int, first: int):
        self.data[winner, first].append(sample)
        self._cap()

    def add_game(self, history: list, winner: int, first: int):
        self.data[winner, first] += history
        self._cap()

    def _cap(self):
        max_bucket_size = self.max_size // 4
        for w, f, data in self.iter_buckets():
            if len(data) > max_bucket_size:
                self.data[w, f] = data[-max_bucket_size:]

    def iter_buckets(self) -> Iterator[tuple[int, int, list]]:
        for w in (0, 1):
            for f in (0, 1):
                yield w, f, self.data[w, f]

    def iter_data(self) -> Iterator[tuple[int, int, list, list, int]]:
        for w, f, data in self.iter_buckets():
            for s, p, r in data:
                yield w, f, s, p, r

    def load(self):
        results = None

        if self.path.exists():
            results = load_pickle(self.path, raise_error=False)
            if not results:
                logger.warn(f"Could not load replay buffer from {self.path}, retrieving backup.")
                path = self.path.with_suffix('.bak')
                results = load_pickle(path)

        if results:
            self._results = results

    def save(self):
        if self.path.exists():
            shutil.copy(self.path, self.path.with_suffix('.bak'))
        dump_pickle(self._results, self.path)

    def pretty_save(self):
        text = ''
        for w, f, data in self.iter_buckets():
            text += f'{w=}, {f=}\n'
            text += '\n'.join(map(str, data))
            text += '\n\n'
        with open(self.path.with_suffix('.txt'), 'w') as f:
            f.write(text)

    def clear(self):
        self._results = dict_factory()
        for w in (0, 1):
            for f in (0, 1):
                self.data[w, f] = []

    def get_winners(self) -> list[int]:
        winners = []
        for w, f, s, p, _ in self.iter_data():
            if p is None:
                s = State.from_list(s)
                assert s.winner is not None
                winners.append(s.winner)
        return winners

    def plot_win_rate(self, avg_over: int = 100):
        winners = self.get_winners()
        avg_over = min(avg_over, len(winners))
        winners = np.array(winners)
        win_rate = 1 - winners
        win_rate = win_rate[-(len(win_rate) // avg_over * avg_over):]
        win_rate = win_rate.reshape(-1, avg_over).mean(axis=1)
        plt.figure()
        plt.plot(win_rate)
        plt.show()
        logger.info(f"Win rate in replay buffer: {win_rate.mean():.0%}")

    def get_diversity_ratio(self, epoch=0) -> float:
        unique_hashes = set()
        for w, f, s, _, _ in self.iter_data():
            unique_hashes.add(str(s))
        diversity_ratio = len(unique_hashes) / len(self)
        self.diversity_history[epoch] = diversity_ratio
        logger.info(f"State uniqueness in replay buffer: {diversity_ratio:.0%}")
        return diversity_ratio

    def filter(self, state: State | list) -> list:
        if isinstance(state, State):
            state = state.to_list()
        return [v for v in self.iter_data() if v[2] == state]
