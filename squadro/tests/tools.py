from abc import ABC, abstractmethod
from unittest import TestCase

from squadro.state import State
from squadro.tools.probabilities import set_seed


class ML(TestCase, ABC):
    def setUp(self):
        set_seed()
        self.evaluator = self.get_evaluator()
        self.state = State(advancement=[[1, 8, 3], [1, 2, 4]], cur_player=0)
        self.end_state = State(advancement=[[8, 8, 3], [1, 2, 4]], cur_player=0)

    @abstractmethod
    def get_evaluator(self):
        """Each child class must implement this to create its specific evaluator"""
        pass

    def test_game_over(self):
        state = self.end_state
        value = self.evaluator.get_value(state)
        self.assertEqual(1, value)

        state.cur_player = 1
        value = self.evaluator.get_value(state)
        self.assertEqual(-1, value)
