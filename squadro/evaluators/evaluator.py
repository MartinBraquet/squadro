import json
import os
from abc import ABC, abstractmethod
from multiprocessing.managers import DictProxy, ArrayProxy  # noqa
from pathlib import Path

import numpy as np
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


class QLearningEvaluator(Evaluator):
    """
    Evaluate a state using a Q-lookup table.

    TODO: should an Evaluator instance be specific to n_pawns or should it be generic for any number
     of pawns like all agents and other evaluators?
    """
    _Q = {}

    def __init__(self, model_path=None, n_pawns=None):
        self.n_pawns = n_pawns or 3
        self.model_path = Path(model_path or DATA_PATH / f"q_table_{self.n_pawns}.json")

    @classmethod
    def reload(cls):
        cls._Q = {}

    @property
    def key(self):
        return str(self.model_path)

    @property
    def Q(self):  # noqa
        if self._Q.get(self.key) is None:
            if os.path.exists(self.key):
                if self.key.endswith(".json"):
                    self._Q[self.key] = json.load(open(self.key, 'r'))
                elif self.key.endswith(".npy"):
                    self._Q[self.key] = np.load(self.key, allow_pickle=True)
                logger.debug(f"Using Q table at {self.key}")
            else:
                if self.key.endswith(".json"):
                    self._Q[self.key] = {}
                elif self.key.endswith(".npy"):
                    shape = get_grid_shape(self.n_pawns)
                    length = np.ravel_multi_index([s - 1 for s in shape], dims=shape)
                    self._Q[self.key] = np.zeros(length, dtype=np.float32)
                logger.warn(f"No file at {self.key}, creating new Q table")
        return self._Q[self.key]

    def dump(self, model_path=None):
        model_path = model_path or self.model_path
        if isinstance(self.Q, dict | DictProxy):
            q = self.Q
            if isinstance(q, DictProxy):
                q = q._getvalue()
            json.dump(q, open(model_path, 'w'), indent=4)
        elif isinstance(self.Q, np.ndarray | ArrayProxy):
            q = self.Q
            if isinstance(q, ArrayProxy):
                q = q._getvalue()
            np.save(model_path, q, allow_pickle=True)

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    def get_value(self, state: State) -> float:
        if state.game_over():
            return 1 if state.winner == state.cur_player else -1

        state_id = self.get_id(state)
        if isinstance(self.Q, np.ndarray | ArrayProxy):
            return self.Q[state_id]
        else:
            return self.Q.get(state_id, 0)

    def get_id(self, state: State):
        if isinstance(self.Q, np.ndarray | ArrayProxy):
            return state_to_index(state)
        else:
            return f'{state.get_advancement()}, {state.cur_player}'

    def set_dict(self, d: DictProxy | dict | np.ndarray) -> None:
        self._Q[self.key] = d


def get_grid_shape(n_pawns: int):
    return [2 * (n_pawns + 1) + 1] * n_pawns * 2 + [2]


def state_to_index(state: State):
    dims = get_grid_shape(state.n_pawns)
    x, y = state.get_advancement()
    return np.ravel_multi_index(x + y + [state.cur_player], dims=dims)


def index_to_state(index: int, n_pawns: int):
    shape = get_grid_shape(n_pawns)
    return np.unravel_index(index, shape=shape)
