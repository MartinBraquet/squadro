from abc import ABC, abstractmethod
from multiprocessing.managers import DictProxy, ArrayProxy  # noqa

import numpy as np
from numpy.typing import NDArray

from squadro.state import State


class Evaluator(ABC):
    """
    Base class for state evaluation.
    """

    @abstractmethod
    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        """
        Evaluate the current state (value and policy), according to the current player.

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

    def get_value(self, state: State) -> float:
        """
        Get the value for the given state.
        """
        return self.evaluate(state)[1]

    @classmethod
    def reload(cls):
        ...
