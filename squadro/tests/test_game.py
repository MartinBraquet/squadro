import random
from unittest import TestCase

import numpy as np

from squadro.game import Game


class TestGame(TestCase):
    def setUp(self):
        np.random.seed(42)
        random.seed(42)

    def test_game(self):
        first = 0
        game = Game(n_pawns=5, agent_0='random', agent_1='random', first=first)
        action_history = game.run()
        self.assertEqual(game.first, first)
        self.assertListEqual(
            action_history,
            [0, 1, 1, 4, 4, 0, 0, 1, 4, 4, 1, 4, 0, 3, 2, 1, 0, 3, 2, 4, 0, 4, 3, 4, 4, 4, 0, 1, 0,
             0, 2, 2, 2, 1, 0, 1, 4, 2, 3, 1, 3, 2, 2, 1, 4, 1, 2, 3, 4, 2, 2, 0, 1, 3, 4, 3, 3, 3,
             4, 0, 4, 2, 3, 0, 1, 0, 4, 0, 3, 0, 3, 0, 3, 0, 1, 2, 1, 2, 4, 0]
        )
        self.assertEqual(game.winner, 0)

    def test_game_3_pawns(self):
        game = Game(n_pawns=3, agent_0='random', agent_1='random')
        action_history = game.run()
        # print(action_history)
