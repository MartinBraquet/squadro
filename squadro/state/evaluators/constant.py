import numpy as np
from numpy._typing import NDArray

from squadro.state.evaluators.base import Evaluator
from squadro.state.state import State


class ConstantEvaluator(Evaluator):
    """
    Evaluate a state as a constant value.
    """

    def __init__(self, constant: float = 0):
        self.constant = constant

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        return p, self.constant
