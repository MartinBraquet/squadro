import json
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from squadro.evaluators.evaluator import QLearningEvaluator
from squadro.state import State
from squadro.tools.probabilities import set_seed


class TestQLearningEvaluator(TestCase):
    def setUp(self):
        set_seed()
        self.evaluator = QLearningEvaluator()

    def test_eval(self):
        state = State(advancement=[[1, 2, 3], [1, 2, 4]], cur_player=0)
        with (
            TemporaryDirectory() as model_path,
            patch.object(self.evaluator, 'model_path', model_path),
        ):
            json.dump(
                {'[[1, 2, 3], [1, 2, 4]], 0': .14},
                open(f"{model_path}/model_3.json", 'w')
            )
            p, value = self.evaluator.evaluate(state)
        self.assertEqual(.14, value)
        np.testing.assert_equal(np.ones(3) / 3, p)

    def test_game_over(self):
        state = State(advancement=[[8, 8, 3], [1, 2, 4]], cur_player=0)
        value = self.evaluator.get_value(state)
        self.assertEqual(1, value)

        state.cur_player = 1
        value = self.evaluator.get_value(state)
        self.assertEqual(-1, value)
