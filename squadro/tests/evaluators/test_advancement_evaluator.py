import random
from unittest import TestCase

import numpy as np

from squadro.evaluators.evaluator import AdvancementEvaluator
from squadro.state import State


class TestAdvancementEvaluator(TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)

    def test_eval(self):
        state = State(first=0, n_pawns=3)
        state.set_from_advancement([[1, 2, 3], [1, 2, 4]])
        evaluator = AdvancementEvaluator()
        p, value = evaluator.evaluate(state)
        self.assertEqual((2 + 3 - 2 - 4) / 16, value)
        np.testing.assert_equal(np.ones(3) / 3, p)
