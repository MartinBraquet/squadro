import tempfile
from pathlib import Path
from time import sleep
from unittest import TestCase
from unittest.mock import patch

import pytest

from squadro.agents.random_agent import RandomAgent
from squadro.core.game import Game, GameFromState
from squadro.core.state import State
from squadro.tools.probabilities import set_seed


class TestGame(TestCase):
    def setUp(self):
        set_seed(42)

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
        self.assertEqual(game.winner, 1)

    def test_game_3_pawns(self):
        game = Game(n_pawns=3, agent_0='random', agent_1='random')
        game.run()

    def test_game_from_state_initialization(self):
        state = State(n_pawns=4, first=1)
        game = GameFromState(state=state, agent_0="random", agent_1="random")
        self.assertEqual(game.first, 1)
        self.assertEqual(game.n_pawns, 4)
        self.assertEqual(game.winner, None)

    def test_game_from_state_action_history(self):
        state = State(n_pawns=3, first=0)
        game = GameFromState(state=state, agent_0="random", agent_1="random")
        self.assertListEqual(game.action_history, [])

    @patch.object(RandomAgent, "get_action", lambda *a, **kw: sleep(2))
    @pytest.mark.slow
    def test_time_out(self):
        game = Game(time_out=1)
        game.run()
        self.assertEqual(game.first, game.state.timeout_player)

    def test_time_outs(self):
        time_out = 10
        game = Game(time_out=time_out)
        game.run()
        self.assertLess(game.times_left[0], time_out)
        self.assertLess(game.times_left[1], time_out)

    def test_to_dict(self):
        game = Game(n_pawns=3)
        self.assertEqual(game.to_dict(), {
            'action_history': [],
            'agent_0': 'random',
            'agent_1': 'random',
            'state': {'first': 0, 'n_pawns': 3},
            'winner': None
        })

    def test_save_and_load_results(self):
        game = Game(n_pawns=3)
        game.run()
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_file = Path(temp_dir) / "test_results.json"
            game.to_file(tmp_file)
            game_loaded = Game.from_file(tmp_file)
        self.assertEqual(game, game_loaded)
