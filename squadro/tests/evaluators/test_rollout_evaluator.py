import random
from unittest import TestCase

import numpy as np

from squadro.evaluators.evaluator import RolloutEvaluator
from squadro.state import State


class TestRolloutEvaluator(TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)

    def test_eval(self):
        state = State(first=0, n_pawns=3)
        state.set_from_advancement([[1, 2, 3], [1, 2, 4]])
        evaluator = RolloutEvaluator()
        p, value = evaluator.evaluate(state)
        self.assertEqual(-1, value)
        np.testing.assert_equal(np.ones(3) / 3, p)
