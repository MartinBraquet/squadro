import random
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest

from squadro.animation.animated_game import GamePlay
from squadro.tools.constants import DefaultParams


def _handle_game_over(*args, **kwargs):
    raise SystemExit()


class TestAnimation(TestCase):
    def setUp(self):
        np.random.seed(42)
        random.seed(42)

    @pytest.mark.slow
    @patch.object(GamePlay, '_handle_game_over', _handle_game_over)
    def test_real_time_animated_game(self, *args):
        with DefaultParams.update(max_time_per_move_real_time=.001):
            GamePlay(agent_0='random', agent_1='best', n_pawns=3).run()
