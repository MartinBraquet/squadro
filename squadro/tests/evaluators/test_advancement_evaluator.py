from unittest import TestCase

import numpy as np

from squadro.state.evaluators.advancement import AdvancementEvaluator
from squadro.state.state import State
from squadro.tools.probabilities import set_seed


class TestAdvancementEvaluator(TestCase):
    def setUp(self):
        set_seed(0)

    def test_eval(self):
        state = State(first=0, n_pawns=3)
        state.set_from_advancement([[1, 2, 3], [1, 2, 4]])
        evaluator = AdvancementEvaluator()
        p, value = evaluator.evaluate(state)
        self.assertEqual((2 + 3 - 2 - 4) / 16, value)
        np.testing.assert_equal(np.ones(3) / 3, p)
