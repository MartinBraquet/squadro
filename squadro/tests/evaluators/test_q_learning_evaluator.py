import json
import random
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from squadro.evaluators.evaluator import QLearningEvaluator
from squadro.state import State


class TestQLearningEvaluator(TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)

    def test_eval(self):
        state = State(first=0, n_pawns=3)
        state.set_from_advancement([[1, 2, 3], [1, 2, 4]])
        evaluator = QLearningEvaluator()
        with NamedTemporaryFile('w') as f, patch.object(evaluator, 'model_path', f.name):
            json.dump({'[[1, 2, 3], [1, 2, 4]], 0': .14}, open(f.name, 'w'))
            p, value = evaluator.evaluate(state)
        self.assertEqual(.14, value)
        np.testing.assert_equal(np.ones(3) / 3, p)

    def test_game_over(self):
        state = State(first=0, n_pawns=3)
        state.set_from_advancement([[8, 8, 3], [1, 2, 4]])
        evaluator = QLearningEvaluator()
        value = evaluator.get_value(state)
        self.assertEqual(1, value)

        state.cur_player = 1
        value = evaluator.get_value(state)
        self.assertEqual(-1, value)
