import json
import random
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from squadro import Game
from squadro.state import State
from squadro.training.q_learning import QLearningTrainer


def run(game: Game):
    state = State(game.n_pawns)
    game.state = state.copy()
    game.state.set_from_advancement([[0, 0, 0], [8, 8, 0]])
    game.state_history = [state, state.get_next_state(0), game.state]


class TestQLearningTrainer(TestCase):
    def setUp(self):
        np.random.seed(42)
        random.seed(42)

    @patch.object(Game, 'run', run)
    def test_from_scratch(self):
        with TemporaryDirectory() as model_path:
            trainer = QLearningTrainer(
                n_pawns=3,
                eval_steps=2,
                eval_interval=3,
                n_steps=4,
                lr=.2,
                gamma=.95,
                model_path=model_path,
            )
            trainer.run()
            q = json.load(open(f"{model_path}/model_3.json"))
            self.assertEqual({
                '[[0, 0, 0], [0, 0, 0]], 0': -0.09386000000000001,
                '[[1, 0, 0], [0, 0, 0]], 1': 0.342,
                '[[0, 0, 0], [8, 8, 0]], 0': -1,
            },
                q
            )

    @patch.object(Game, 'run', run)
    def test_from_file(self):
        with TemporaryDirectory() as model_path:
            key = '[[1, 2, 3], [4, 5, 6]], 0'
            value = 0.42
            json.dump({key: value}, open(f"{model_path}/model_3.json", mode='w'))
            trainer = QLearningTrainer(
                n_pawns=3,
                eval_steps=2,
                eval_interval=3,
                n_steps=4,
                lr=.2,
                gamma=.95,
                model_path=model_path,
            )
            trainer.run()
            q = json.load(open(f"{model_path}/model_3.json"))
            self.assertEqual(value, q[key])
            self.assertGreater(len(q), 1)
