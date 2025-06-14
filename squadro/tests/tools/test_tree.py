from unittest import TestCase

from squadro.state.state import State, get_next_state
from squadro.tools.probabilities import set_seed
from squadro.tools.tree import Node, Stats, Edge


class TestNode(TestCase):
    def setUp(self):
        set_seed(0)

    def test_node(self):
        state = State(first=0, n_pawns=3)
        node = Node(state)
        self.assertEqual(state, node.state)
        self.assertEqual(0, node.depth)
        self.assertTrue(node.is_leaf)
        self.assertEqual(0, node.player_turn)


class TestStats(TestCase):
    def setUp(self):
        set_seed(0)

    def test(self):
        stats = Stats(prior=.5)
        self.assertEqual({'N': 0, 'W': 0, 'Q': 0, 'P': .5}, stats.dict())
        stats.update(1.4)
        self.assertEqual({'N': 1, 'W': 1.4, 'Q': 1.4, 'P': .5}, stats.dict())
        stats.update(1.6)
        self.assertEqual({'N': 2, 'W': 3.0, 'Q': 1.5, 'P': .5}, stats.dict())


class TestEdges(TestCase):
    def setUp(self):
        set_seed(0)

    def test(self):
        action = 0
        state = State(first=0, n_pawns=3)
        in_node = Node(state)
        out_node = Node(get_next_state(state, action=action))
        self.assertEqual(out_node.state.get_advancement(), [[1, 0, 0], [0, 0, 0]])
        edge = Edge(in_node, out_node, action=action, prior=.5)
        self.assertEqual(0, edge.player_turn)
