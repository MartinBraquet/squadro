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
    """
    Evaluate a state according to the advancement heuristic.
    """
    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = np.ones(state.n_pawns) / state.n_pawns
        value = evaluate_advancement(state=state)
        return p, value


class ConstantEvaluator(Evaluator):
    """
    Evaluate a state as a constant value.
    """
    def __init__(self, constant: float = 0):
        self.constant = constant

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = np.ones(state.n_pawns) / state.n_pawns
        return p, self.constant


class RolloutEvaluator(Evaluator):
    """
    Evaluate a state using random playouts until the end of the game.
    """
    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = np.ones(state.n_pawns) / state.n_pawns
        value = self.get_value(state)
        return p, value

    @staticmethod
    def get_value(state: State) -> float:
        cur_player = state.cur_player
        while not state.game_over():
            action = state.get_random_action()
            state = state.get_next_state(action)
        return 1 if state.winner == cur_player else -1
