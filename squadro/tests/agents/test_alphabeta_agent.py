import random
from time import sleep
from unittest import TestCase
from unittest.mock import patch

from squadro import minimax
from squadro.agents.alphabeta_agent import (
    AlphaBetaAdvancementAgent,
    AlphaBetaRelativeAdvancementAgent, AlphaBetaAgent, AlphaBetaAdvancementDeepAgent,
)
from squadro.agents.random_agent import RandomAgent
from squadro.squadro_state import SquadroState


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

    def cutoff(self, state: SquadroState, depth: int):
        return depth >= self.MAX_DEPTH

    def evaluate(self, state: SquadroState):
        self.evaluations += 1
        return random.random()


class TestAlphaBeta(TestCase):
    def setUp(self):
        random.seed(1)

    def test_get_action(self):
        agent = RandomAlphaBetaAgent(pid=0)
        state = SquadroState(first=0, n_pawns=5)
        action = agent.get_action(state)

        # Less than n_pawns^MAX_DEPTH because of pruning
        self.assertEqual(875, agent.evaluations)

        self.assertEqual(3, action)

        state.apply_action(action)

        action = agent.get_action(state)

        self.assertEqual(1, action)

    def test_successors(self):
        self.agent = AlphaBetaAdvancementAgent(pid=0)
        self.state = SquadroState(n_pawns=3, first=0)
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
        self.state = SquadroState(first=0, n_pawns=3)

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
        self.state = SquadroState(first=0, n_pawns=3)

    def test_evaluate(self):
        self.state.pos[0] = [4, 2, 1]
        self.state.returning[0] = [True, False, False]
        self.state.finished[0] = [True, False, False]
        self.state.pos[1] = [4, 4, 0]
        self.assertEqual(8 + 2 + 3 - 4, self.agent.evaluate(self.state))


class TestAdvancementDeep(TestCase):
    def setUp(self):
        self.agent = AlphaBetaAdvancementDeepAgent(pid=0, max_depth=5)
        self.state = SquadroState(first=0, n_pawns=3)

    def test_evaluate(self):
        self.state.pos[0] = [4, 2, 1]
        self.state.returning[0] = [True, False, False]
        self.state.finished[0] = [True, False, False]
        self.state.pos[1] = [4, 4, 0]
        self.assertEqual(8 + 3 - 4, self.agent.evaluate(self.state))

    def test_get_action(self):
        action = self.agent.get_action(self.state)

        # Hard to produce consistent test results because of the time dependence of the algorithm
        self.assertEqual(2, action)

        self.assertGreater(self.agent.depth, 0)

    @patch.object(minimax, 'search', lambda *a, **kw: sleep(.01) or 2)
    @patch.object(SquadroState, 'get_random_action', return_value=-1)
    def test_skip_unfinished_depth(self, *args, **kwargs):
        """
        Test that the search skips the last minimax result if it finished due to timeout, which
        prevented it from exploring all the leaf nodes at that depth, and hence does not guarantee
        that the best move is found.
        """
        action = self.agent.get_action(self.state)
        self.assertEqual(-1, action)

    def test_minimax_timeout(self, *args, **kwargs):
        """
        Make sure minimax always explores at least the children of the root node, even if too long
        to finish on time. Otherwise, it can't output an action.
        """
        with patch.object(self.agent, 'max_time', 1e-9):
            action = self.agent.get_action(self.state)
        self.assertTrue(self.state.is_action_valid(action))
