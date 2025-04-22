import json
import os
from abc import ABC, abstractmethod
from multiprocessing.managers import DictProxy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from squadro.state import State
from squadro.tools.constants import DATA_PATH, DefaultParams
from squadro.tools.evaluation import evaluate_advancement


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
    """
    _Q = {}

    def __init__(self, model_path=None, n_pawns=None):
        n_pawns = n_pawns or DefaultParams.n_pawns
        self.model_path = Path(model_path or DATA_PATH / f"q_table_{n_pawns}.json")

    @classmethod
    def reload(cls):
        cls._Q = {}

    @property
    def key(self):
        return str(self.model_path)

    @property
    def Q(self):  # noqa
        if self._Q.get(self.key) is None:
            if os.path.exists(self.model_path):
                self._Q[self.key] = json.load(open(self.model_path, 'r'))
            else:
                self._Q[self.key] = {}
        return self._Q[self.key]

    def dump(self, model_path=None):
        model_path = model_path or self.model_path
        json.dump(dict(self.Q), open(model_path, 'w'), indent=4)

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    def get_value(self, state: State) -> float:
        if state.game_over():
            return 1 if state.winner == state.cur_player else -1

        state_id = self.get_id(state)
        return self.Q.get(state_id, 0)

    @classmethod
    def get_id(cls, state: State) -> str:
        return f'{state.get_advancement()}, {state.cur_player}'

    def set_dict(self, d: DictProxy | dict) -> None:
        self._Q[self.key] = d
