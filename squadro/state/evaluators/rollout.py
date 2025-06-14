import numpy as np
from numpy._typing import NDArray

from squadro.state.evaluators.evaluator import Evaluator
from squadro.state.state import State


class RolloutEvaluator(Evaluator):
    """
    Evaluate a state using random playouts until the end of the game.
    """

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    @staticmethod
    def get_value(state: State, **kwargs) -> float:
        cur_player = state.cur_player
        while not state.game_over():
            action = state.get_random_action()
            state = state.get_next_state(action)
        return 1 if state.winner == cur_player else -1
