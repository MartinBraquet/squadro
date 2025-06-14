import numpy as np
from numpy.typing import NDArray

from squadro.state.evaluators.base import Evaluator
from squadro.state.state import State
from squadro.tools.evaluation import evaluate_advancement


class AdvancementEvaluator(Evaluator):
    """
    Evaluate a state according to the advancement heuristic.
    """

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    @staticmethod
    def get_value(state: State, **kwargs) -> float:
        return evaluate_advancement(state=state)
