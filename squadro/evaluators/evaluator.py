from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from squadro.state import State
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


class AdvancementEvaluator(Evaluator):
    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = np.ones(state.n_pawns) / state.n_pawns
        value = evaluate_advancement(
            state=state,
            player_id=state.cur_player,
        )
        return p, value


class ConstantEvaluator(Evaluator):
    def __init__(self, constant: float = 0):
        self.constant = constant

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = np.ones(state.n_pawns) / state.n_pawns
        return p, self.constant


class RandomPlayoutEvaluator(Evaluator):
    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = np.ones(state.n_pawns) / state.n_pawns
        value = ...
        return p, value
