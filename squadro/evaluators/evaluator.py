import json
import os
from abc import ABC, abstractmethod
from multiprocessing.managers import DictProxy, ArrayProxy  # noqa
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from squadro.state import State
from squadro.tools.constants import DATA_PATH
from squadro.tools.evaluation import evaluate_advancement
from squadro.tools.log import logger


class Evaluator(ABC):
    """
    Base class for state evaluation.
    """

    @abstractmethod
    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        """
        Evaluate the current state (Q value and policy), according to the current player.

        Args:
            state: The current game state to evaluate

        Returns:
            A tuple containing:
                - NDArray[np.float64]: Probability distribution over possible actions (policy)
                - float: Value estimation for the current state from the perspective of the player
                playing the next move at that state.
        """
        ...

    @staticmethod
    def get_policy(state: State) -> np.ndarray:
        """
        Get the policy for the given state.
        """
        return np.ones(state.n_pawns) / state.n_pawns

    @staticmethod
    def get_value(state: State) -> float:
        """
        Get the value for the given state.
        """
        raise NotImplementedError

    @classmethod
    def reload(cls):
        ...


class AdvancementEvaluator(Evaluator):
    """
    Evaluate a state according to the advancement heuristic.
    """

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    @staticmethod
    def get_value(state: State) -> float:
        return evaluate_advancement(state=state)


class ConstantEvaluator(Evaluator):
    """
    Evaluate a state as a constant value.
    """

    def __init__(self, constant: float = 0):
        self.constant = constant

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        return p, self.constant


class RolloutEvaluator(Evaluator):
    """
    Evaluate a state using random playouts until the end of the game.
    """

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    @staticmethod
    def get_value(state: State) -> float:
        cur_player = state.cur_player
        while not state.game_over():
            action = state.get_random_action()
            state = state.get_next_state(action)
        return 1 if state.winner == cur_player else -1


def get_grid_shape(n_pawns: int):
    return [2 * (n_pawns + 1) + 1] * n_pawns * 2 + [2]


def state_to_index(state: State):
    dims = get_grid_shape(state.n_pawns)
    x, y = state.get_advancement()
    return np.ravel_multi_index(x + y + [state.cur_player], dims=dims)


def index_to_state(index: int, n_pawns: int):
    shape = get_grid_shape(n_pawns)
    return np.unravel_index(index, shape=shape)


class _RLEvaluator(Evaluator, ABC):
    _default_dir = 'default'

    def __init__(self, model_path: str | Path = None, dtype='json'):
        """
        :param model_path: Path to the directory where the model is stored.
        """
        self.model_path = Path(model_path or DATA_PATH / self._default_dir)
        self.dtype = dtype

    @classmethod
    def reload(cls):
        cls._Q = {}

    def get_filepath(self, n_pawns: int, model_path=None) -> str:
        model_path = Path(model_path or self.model_path)
        return str(model_path / f"model_{n_pawns}.{self.dtype}")

    def clear(self):
        self._Q[self.dir_key] = {}

    @property
    def dir_key(self):
        return str(self.model_path)

    @property
    def models(self):
        """
        :return: A dictionary mapping pawn numbers to Q tables.
        """
        if self.dir_key not in self._Q:
            self._Q[self.dir_key] = {}
        return self._Q[self.dir_key]

    def set_model(
        self,
        d: DictProxy | dict | np.ndarray | ArrayProxy | torch.nn.Module,
        n_pawns: int,
    ) -> None:
        self.models[n_pawns] = d

    def dump(self, model_path: str | Path = None):
        model_path = str(model_path or self.model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for n_pawns, model in self.models.items():
            filepath = self.get_filepath(n_pawns, model_path=model_path)
            self._dump(model, filepath)

    @abstractmethod
    def _dump(self, model, filepath: str):
        ...


class QLearningEvaluator(_RLEvaluator):
    """
    Evaluate a state using a Q-lookup table.

     _Q is a cache for all Q-tables.
     _Q['/path/dir'][3] is the Q-table in '/path/dir' for 3 pawns (stared in file '/path/dir/model_3.json')
    """
    _Q = {}
    _default_dir = 'q_learning'

    @property
    def is_json(self):
        return self.dtype == 'json'

    def get_Q(self, n_pawns: int) -> dict:  # noqa
        if self.models.get(n_pawns) is None:
            filepath = self.get_filepath(n_pawns)
            if os.path.exists(filepath):
                if self.is_json:
                    self.models[n_pawns] = json.load(open(filepath, 'r'))
                else:
                    self.models[n_pawns] = np.load(filepath, allow_pickle=True)
                logger.info(f"Using Q table at {filepath}")
            else:
                if self.is_json:
                    self.models[n_pawns] = {}
                else:
                    shape = get_grid_shape(n_pawns)
                    length = np.ravel_multi_index([s - 1 for s in shape], dims=shape)
                    self.models[n_pawns] = np.zeros(length, dtype=np.float32)
                logger.warn(f"No file at {filepath}, creating new Q table")

        return self.models[n_pawns]

    def _dump(self, model, filepath: str):
        if isinstance(model, (DictProxy, ArrayProxy)):
            model = model._getvalue()
        if isinstance(model, dict):
            json.dump(model, open(filepath, 'w'), indent=4)
        elif isinstance(model, np.ndarray):
            np.save(filepath, model, allow_pickle=True)

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    def get_value(self, state: State, check_game_over: bool = True) -> float:
        if check_game_over and state.game_over():
            return 1 if state.winner == state.cur_player else -1

        q = self.get_Q(state.n_pawns)
        state_id = self.get_id(state)
        if self.is_json:
            return q.get(state_id, 0)
        else:
            return q[state_id]

    def get_id(self, state: State):
        if self.is_json:
            return f'{state.get_advancement()}, {state.cur_player}'
        else:
            return state_to_index(state)


class DeepQLearningEvaluator(_RLEvaluator):
    """
    Evaluate a state using a deep neural network.

     _Q is a cache for all neural networks.
     _Q['/path/dir'][3] is the neural network in '/path/dir' for 3 pawns (stared in file '/path/dir/model_3.json')
    """

    _Q = {}
    _default_dir = 'deep_q_learning'

    def __init__(self, **kwargs):
        """
        :param model_path: Path to the directory where the model is stored.
        """
        kwargs.setdefault('dtype', 'pt')
        super().__init__(**kwargs)

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        pass

    def get_model(self, n_pawns: int) -> torch.nn.Module:
        if self.models.get(n_pawns) is None:
            filepath = self.get_filepath(n_pawns)
            if os.path.exists(filepath):
                self.models[n_pawns] = torch.load(filepath)
                logger.info(f"Using model at {filepath}")
            else:
                self.models[n_pawns] = ...  # TODO: create new model
                logger.warn(f"No file at {filepath}, creating new model")

        return self.models[n_pawns]

    def _dump(self, model, filepath: str):
        torch.save(obj=model, f=filepath)
