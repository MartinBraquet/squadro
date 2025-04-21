import random
from time import time
from unittest import TestCase

import numpy as np

from squadro.agents.best import get_best_real_time_game_agent
from squadro.state import State
from squadro.tools.constants import DefaultParams


class TestBest(TestCase):
    def setUp(self):
        np.random.seed(42)
        random.seed(42)

    def test(self):
        n_pawns = 3
        state = State(n_pawns=n_pawns)
        max_time_per_move = .01
        with DefaultParams.update(max_time_per_move_real_time=max_time_per_move):
            agent = get_best_real_time_game_agent()
            t = time()
            action = agent.get_action(state)
            t = time() - t
        self.assertLess(t, max_time_per_move + 1e3)
        self.assertIn(action, {0, 1, 2})
