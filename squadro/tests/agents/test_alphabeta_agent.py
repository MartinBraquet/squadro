import random
from time import sleep, time
from unittest import TestCase
from unittest.mock import patch

from squadro import minimax
from squadro.agents.alphabeta_agent import (
    AlphaBetaAdvancementAgent,
    AlphaBetaRelativeAdvancementAgent,
    AlphaBetaAgent,
    AlphaBetaAdvancementDeepAgent,
)
from squadro.agents.random_agent import RandomAgent
from squadro.state import State


class RandomAlphaBetaAgent(AlphaBetaAgent):
    """
    Agent performing alpha-beta tree search up to a certain depth.
    State evaluation is done randomly.
    Use it to test the minimax tree search in isolation from other routines called
    inside it, like cutoff and evaluation.
    """
    MAX_DEPTH = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluations = 0

    @classmethod
    def get_name(cls) -> str:
        return "random_alpha_beta"

    def cutoff(self, state: State, depth: int):
        return depth >= self.MAX_DEPTH

    def evaluate(self, state: State):
        self.evaluations += 1
        return random.random()


class TestAlphaBeta(TestCase):
    def setUp(self):
        random.seed(1)

    def test_get_action(self):
        agent = RandomAlphaBetaAgent(pid=0)
        state = State(first=0, n_pawns=5)
        action = agent.get_action(state)

        # Less than n_pawns^MAX_DEPTH because of pruning
        self.assertEqual(875, agent.evaluations)

        self.assertEqual(3, action)

        state.apply_action(action)

        action = agent.get_action(state)

        self.assertEqual(1, action)

    def test_successors(self):
        self.agent = AlphaBetaAdvancementAgent(pid=0)
        self.state = State(n_pawns=3, first=0)
        expected = [
            (0, [[3, 4, 4], [4, 4, 4]]),
            (1, [[4, 1, 4], [4, 4, 4]]),
            (2, [[4, 4, 2], [4, 4, 4]])
        ]
        for i, (a, s) in enumerate(self.agent.successors(self.state)):
            exp = expected[i]
            self.assertEqual(a, exp[0])
            self.assertEqual(s.pos, exp[1])
            self.assertEqual(s.cur_player, 1 - self.state.cur_player)


class TestAdvancement(TestCase):
    def setUp(self):
        self.agent = AlphaBetaAdvancementAgent(pid=0)
        self.state = State(first=0, n_pawns=3)

    def test_cutoff(self):
        cutoff = self.agent.cutoff(self.state, 1)
        self.assertTrue(cutoff)

        cutoff = self.agent.cutoff(self.state, 0)
        self.assertFalse(cutoff)

        agent = RandomAgent()
        while not self.state.game_over():
            action = agent.get_action(self.state)
            self.state.apply_action(action)
        cutoff = self.agent.cutoff(self.state, 0)
        self.assertTrue(cutoff)

    def test_evaluate(self):
        self.assertEqual(0, self.agent.evaluate(self.state))
        self.state.pos[0] = [4, 2, 1]
        self.state.returning[0] = [True, False, False]
        self.state.finished[0] = [True, False, False]
        self.assertEqual(8 + 2 + 3, self.agent.evaluate(self.state))


class TestRelativeAdvancement(TestCase):
    def setUp(self):
        self.agent = AlphaBetaRelativeAdvancementAgent(pid=0)
        self.state = State(first=0, n_pawns=3)

    def test_evaluate(self):
        self.state.pos[0] = [4, 2, 1]
        self.state.returning[0] = [True, False, False]
        self.state.finished[0] = [True, False, False]
        self.state.pos[1] = [4, 4, 0]
        self.assertEqual(8 + 2 + 3 - 4, self.agent.evaluate(self.state))


class TestAdvancementDeep(TestCase):
    def setUp(self):
        self.agent = AlphaBetaAdvancementDeepAgent(pid=0, max_depth=5)
        self.state = State(first=0, n_pawns=3)

    def test_evaluate(self):
        self.state.pos[0] = [4, 2, 1]
        self.state.returning[0] = [True, False, False]
        self.state.finished[0] = [True, False, False]
        self.state.pos[1] = [4, 4, 0]
        self.assertEqual((8 + 3 - 4) / 16, self.agent.evaluate(self.state))

    def test_get_action(self):
        self.agent.max_time_per_move = 1e9
        action = self.agent.get_action(self.state)
        self.assertEqual(2, action)
        self.assertGreater(self.agent.depth, 0)

    @patch.object(minimax, 'search', lambda *a, **kw: sleep(.01) or 2)
    @patch.object(State, 'get_random_action', return_value=-1)
    def test_skip_unfinished_depth(self, *args, **kwargs):
        """
        Test that the search skips the last minimax result if it finished due to timeout, which
        prevented it from exploring all the leaf nodes at that depth, and hence does not guarantee
        that the best move is found.
        """
        action = self.agent.get_action(self.state)
        self.assertEqual(-1, action)

    def test_minimax_short_timeout(self, *args, **kwargs):
        """
        Make sure minimax always explores at least the children of the root node, even if too long
        to finish on time. Otherwise, it can't output an action.
        """
        with patch.object(self.agent, 'max_time_per_move', 1e-9):
            action = self.agent.get_action(self.state)
        self.assertTrue(self.state.is_action_valid(action))

    def test_minimax_timeout(self, *args, **kwargs):
        """
        Make sure iterative depth search is stopped when time runs out
        """
        time_out = .001
        self.agent.max_time_per_move = time_out
        compute_time = time()
        self.agent.get_action(self.state)
        compute_time = time() - compute_time
        self.assertLess(compute_time, time_out + 5e-4)
        self.assertGreater(self.agent.depth, 0)
        self.assertLess(self.agent.depth, self.agent.max_depth)

    def test_favor_winning_move(self, *args, **kwargs):
        """
        Make sure that the agent picks the winning move
        """
        self.agent.max_depth = 1
        state = State(first=0, n_pawns=5)
        state.set_from_advancement([[12, 12, 12, 11, 0], [12, 12, 12, 9, 11]])
        action = self.agent.get_action(state)
        self.assertEqual(3, action)

    def test_winning_state(self, *args, **kwargs):
        """
        Make sure the winning states have infinite value
        """
        state = State(first=0, n_pawns=5)
        zero = [0, 0, 0, 0, 0]
        winning = [12, 12, 12, 12, 0]
        max_value = 1

        state.set_from_advancement([winning, zero])
        value = self.agent.evaluate(state)
        self.assertEqual(max_value, value)

        state.set_from_advancement([zero, winning])
        value = self.agent.evaluate(state)
        self.assertEqual(- max_value, value)

        self.agent.id = 1
        state.set_from_advancement([winning, zero])
        value = self.agent.evaluate(state)
        self.assertEqual(-max_value, value)

        state.set_from_advancement([zero, winning])
        value = self.agent.evaluate(state)
        self.assertEqual(max_value, value)

    def test_all_winning_moves(self, *args, **kwargs):
        """
        Make sure that it returns an action when all the moves are winning
        """
        self.state.set_from_advancement([[0, 0, 8], [7, 7, 8]])
        action = self.agent.get_action(self.state)
        self.assertIn(action, {0, 1})
