import random
from unittest import TestCase

import numpy as np

from squadro.agents.montecarlo_agent import MonteCarloAgent
from squadro.game import Game
from squadro.state import State


class TestMonteCarlo(TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)

    def test_get_action(self):
        agent = MonteCarloAgent(pid=0)
        agent.max_time = 1e9

        state = State(first=0, n_pawns=3)
        for i in range(1, 4):
            state.set_from_advancement([[0, 0, 0], [i] * 3])
            action = agent.get_action(state)
            self.assertEqual(3 - i, action)

    def test_game_ab(self):
        game = Game(n_pawns=4, agent_0='mcts', agent_1='ab_advancement_deep', first=0)
        game.agents[0].mc_steps = 200
        game.run()
        self.assertEqual(game.winner, 0)

    def test_game(self):
        game = Game(n_pawns=4, agent_0='mcts', agent_1='random', first=0)
        game.agents[0].mc_steps = 50
        action_history = game.run()
        self.assertEqual(
            [1, 0, 3, 0, 3, 0, 0, 1, 0, 3, 0, 2, 0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 3, 1, 1, 1, 2, 2,
             3, 2, 0, 2, 1, 2, 2, 2],
            action_history
        )
        self.assertEqual(game.winner, 0)
