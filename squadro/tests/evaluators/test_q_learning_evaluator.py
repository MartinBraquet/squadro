import json
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from squadro.state.evaluators.rl import QLearningEvaluator
from squadro.state.state import State
from squadro.tests.evaluators.tools import ML


class TestQLearningEvaluator(ML):
    def get_evaluator(self):
        return QLearningEvaluator()

    def test_eval(self):
        state = State(advancement=[[1, 2, 3], [1, 2, 4]], cur_player=0)
        with (
            TemporaryDirectory() as model_path,
            patch.object(self.evaluator, '_model_path', model_path),
        ):
            json.dump(
                {'[[1, 2, 3], [1, 2, 4]], 0': .14},
                open(f"{model_path}/model_3.json", 'w')
            )
            p, value = self.evaluator.evaluate(state)
        self.assertEqual(.14, value)
        np.testing.assert_equal(np.ones(3) / 3, p)
