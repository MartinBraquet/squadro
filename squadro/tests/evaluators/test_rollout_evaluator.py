from unittest import TestCase

import numpy as np

from squadro.core.state import State
from squadro.evaluators.rollout import RolloutEvaluator
from squadro.tools.probabilities import set_seed


class TestRolloutEvaluator(TestCase):
    def setUp(self):
        set_seed(0)

    def test_eval(self):
        state = State(first=0, n_pawns=3)
        state.set_from_advancement([[1, 2, 3], [1, 2, 4]])
        evaluator = RolloutEvaluator()
        p, value = evaluator.evaluate(state)
        self.assertEqual(-1, value)
        np.testing.assert_equal(np.ones(3) / 3, p)
