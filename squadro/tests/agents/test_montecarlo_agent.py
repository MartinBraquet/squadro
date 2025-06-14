import copy
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from squadro.agents.montecarlo_agent import MonteCarloAdvancementAgent, MCTS, \
    MonteCarloRolloutAgent, MonteCarloQLearningAgent, MonteCarloDeepQLearningAgent
from squadro.core.game import Game
from squadro.state.evaluators.advancement import AdvancementEvaluator
from squadro.state.evaluators.constant import ConstantEvaluator
from squadro.state.state import State
from squadro.tools.constants import inf
from squadro.tools.probabilities import set_seed
from squadro.tools.tree import Node


class TestMonteCarlo(TestCase):
    def setUp(self):
        set_seed()
        self.state = State(first=0, n_pawns=3)

    def test_get_action_tricky(self):
        self.state.set_from_advancement([[0, 4, 8], [5, 2, 8]])
        agent = MonteCarloAdvancementAgent(
            pid=0,
            max_time_per_move=1e9,
            mcts_kwargs=dict(max_steps=50),

        )
        action = agent.get_action(self.state)
        self.assertEqual(0, action)

    def test_get_action(self):
        agent = MonteCarloAdvancementAgent(
            pid=0,
            max_time_per_move=1e9,
            mcts_kwargs=dict(max_steps=50),
        )
        for i in range(1, 4):
            self.state.set_from_advancement([[0, 0, 0], [i] * 3])
            action = agent.get_action(self.state)
            self.assertEqual(3 - i, action)

    def test_game_ab(self):
        agent = MonteCarloAdvancementAgent(mcts_kwargs=dict(max_steps=200, stochastic_moves=10))
        game = Game(n_pawns=4, agent_0=agent, agent_1='relative_advancement', first=0)
        game.run()
        self.assertEqual(game.winner, 0)

    def test_p_uct(self):
        agent = MonteCarloAdvancementAgent(
            mcts_kwargs=dict(
                uct=1,
                method='p_uct',
                max_steps=10,
            ),
            max_time_per_move=1e9,
        )
        game = Game(agent_0=agent, agent_1='random', n_pawns=3, first=0)
        action_history = game.run()
        self.assertEqual(game.winner, 0)
        self.assertEqual(
            [0, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            action_history
        )

    def test_uct(self):
        agent = MonteCarloAdvancementAgent(
            mcts_kwargs=dict(
                uct=1,
                method='uct',
                max_steps=10,
            ),
            max_time_per_move=1e9,
        )
        game = Game(agent_0=agent, agent_1='random', n_pawns=3, first=0)
        action_history = game.run()
        self.assertEqual(game.winner, 0)
        self.assertEqual(
            [0, 2, 1, 1, 0, 2, 1, 2, 0, 1, 1, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 1, 1, 2, 1, 2, 0, 2],
            action_history
        )

    def test_biased_uct(self):
        agent = MonteCarloAdvancementAgent(
            max_time_per_move=1e9,
            mcts_kwargs=dict(
                uct=1,
                method='biased_uct',
                max_steps=10,
                tau=1,
                epsilon_action=.3,
                p_mix=.2,
            )
        )
        game = Game(agent_0=agent, agent_1='random', n_pawns=3, first=0)
        action_history = game.run()
        self.assertEqual(game.winner, 0)
        self.assertEqual(
            [1, 2, 1, 1, 2, 0, 1, 0, 1, 1, 2, 2, 2, 0, 2, 0, 2],
            action_history
        )

    def test_heuristic(self):
        """
        Test that the heuristic for biased UCT is correct.
        """
        action = 0
        state = self.state
        state.set_from_advancement([[0, 0, 1], [2, 2, 2]])
        edge = MCTS.get_edge(parent=Node(state), action=action)
        h = MCTS._get_heuristic(edge)
        self.assertEqual((1 + 1 - 2 - 2) / 16, h)

    def test_expand_leaf(self):
        root = Node(self.state)
        probs = np.array([.2, .3, .5])
        mcts = MCTS(root, evaluator=AdvancementEvaluator())
        mcts._expand_leaf(root, probs=probs)
        self.assertEqual(3, len(root.edges))
        self.assertEqual(3, len(root.children))

        edge = root.edges[0]
        self.assertEqual(0, edge.player_turn)
        self.assertEqual(1, edge.out_node.player_turn)
        self.assertEqual(root, edge.in_node)
        self.assertEqual([[1, 0, 0], [0, 0, 0]], edge.out_node.state.get_advancement())
        self.assertEqual(0.2, edge.stats.P)
        self.assertEqual(0, edge.action)

        edge = root.edges[1]
        self.assertEqual(0, edge.player_turn)
        self.assertEqual(1, edge.out_node.player_turn)
        self.assertEqual(root, edge.in_node)
        self.assertEqual([[0, 3, 0], [0, 0, 0]], edge.out_node.state.get_advancement())
        self.assertEqual(0.3, edge.stats.P)
        self.assertEqual(1, edge.action)

        edge = root.edges[2]
        self.assertEqual(0, edge.player_turn)
        self.assertEqual(1, edge.out_node.player_turn)
        self.assertEqual(root, edge.in_node)
        self.assertEqual([[0, 0, 2], [0, 0, 0]], edge.out_node.state.get_advancement())
        self.assertEqual(0.5, edge.stats.P)
        self.assertEqual(2, edge.action)

    def test_expand_leaf_limited_action(self):
        state = self.state
        state.set_from_advancement([[8, 0, 0], [0, 0, 0]])
        root = Node(state)
        mcts = MCTS(root, evaluator=AdvancementEvaluator())
        mcts._expand_leaf(root)
        self.assertEqual(2, len(root.edges))

        edge = root.edges[0]
        self.assertEqual([[8, 3, 0], [0, 0, 0]], edge.out_node.state.get_advancement())
        self.assertEqual(None, edge.stats.P)
        self.assertEqual(1, edge.action)

    def test_expand_leaf_game_over(self):
        state = self.state
        state.set_from_advancement([[8, 0, 8], [0, 0, 0]])
        root = Node(state)
        mcts = MCTS(root, evaluator=AdvancementEvaluator())
        mcts._expand_leaf(root)
        self.assertEqual([], root.edges)

    @patch.object(MCTS, '_get_heuristic', lambda *a, **kw: .69)
    def test_sim_edge_values(self):
        root = Node(self.state)
        mcts = MCTS(
            root,
            evaluator=AdvancementEvaluator(),
            uct=1.12,
            p_mix=0.217,
            is_training=True,
            a_dirichlet=.8,
        )
        mcts._expand_leaf(root, probs=np.array([.2, .2, .2]))
        root.edges[0].stats.update(8.0)
        root.edges[0].stats.update(6.0)

        # From seeded sample of np.random.dirichlet
        nu = 0.35209437973900337

        mcts.method = 'p_uct'
        values = mcts._get_sim_edge_values(root)
        p = (1 - 0.217) * .2 + 0.217 * nu
        self.assertEqual(
            7 + 1.12 * p * np.sqrt(2) / (1 + 2),
            values[0]
        )

        mcts.method = 'biased_uct'
        values = mcts._get_sim_edge_values(root)
        self.assertEqual(
            7 + 1.12 * .69 / (1 + 2),
            values[0]
        )

        root.edges[1].stats.update(1.0)

        mcts.method = 'uct'
        values = mcts._get_sim_edge_values(root)
        np.testing.assert_almost_equal([
            7 + 1.12 * np.sqrt(np.log(3) / 2) + 0.217 * 0.9345319855570997,
            1 + 1.12 * np.sqrt(np.log(3) / 1) + 0.217 * 0.0542002849217704,
            inf
        ],
            values
        )

        mcts.method = 'abcd'
        with self.assertRaises(ValueError):
            mcts._get_sim_edge_values(root)

    @patch.object(MCTS, '_get_sim_edge_values', lambda *a, **kw: [.2, .5, .3])
    def test_pick_sim_edge(self):
        root = Node(self.state)
        mcts = MCTS(root, evaluator=AdvancementEvaluator())
        mcts._expand_leaf(root)
        edge = mcts._pick_sim_edge(root)
        self.assertEqual(root.edges[1], edge)

    @patch.object(MCTS, '_get_sim_edge_values', lambda *a, **kw: [.2, .5, .3])
    def test_move_to_leaf(self):
        root = Node(self.state)
        mcts = MCTS(root, evaluator=AdvancementEvaluator())

        mcts._expand_leaf(root)
        for edge in root.edges:
            mcts._expand_leaf(edge.out_node)

        node, trajectory = mcts.move_to_leaf()
        self.assertEqual(root.edges[1].out_node.edges[1].out_node, node)
        self.assertEqual([
            root.edges[1],
            root.edges[1].out_node.edges[1]
        ],
            trajectory
        )

    def test_evaluate_leaf(self):
        state = self.state
        state.set_from_advancement([[1, 1, 1], [0, 0, 0]])
        root = Node(state)
        mcts = MCTS(root, evaluator=ConstantEvaluator(42))
        value = mcts.evaluate_leaf(root)
        self.assertEqual(42, value)
        self.assertEqual(3, len(root.edges))

    def test_backfill(self):
        state = self.state
        root = Node(state)
        mcts = MCTS(root, evaluator=ConstantEvaluator(), method='uct')
        for i in range(4):
            mcts.simulate()

        leaf, trajectory = mcts.move_to_leaf()

        mcts.back_fill(
            player_turn=0,
            value=42,
            trajectory=trajectory,
        )
        values = [e.stats.W for e in trajectory]
        self.assertEqual([42, -42], values)

    def test_get_av(self):
        state = self.state
        root = Node(state)
        mcts = MCTS(
            root,
            evaluator=ConstantEvaluator(2),
            method='uct',
            epsilon_action=.3,
            tau=1,
        )
        mcts._expand_leaf(root)
        for i, edge in enumerate(root.edges):
            edge.stats.N = i
            edge.stats.W = i ** 3

        pi, values = mcts.get_av()
        np.testing.assert_almost_equal(np.array([0.0, 1 / 3, 2 / 3]), pi, decimal=7)
        self.assertEqual([0.0, 1.0, 4.0], values.tolist())

    def test_get_av_limited_actions(self):
        state = State.from_list([[[3, 4, 2], [0, 2, 8]], 1, 0])
        root = Node(state)
        mcts = MCTS(
            root,
            evaluator=ConstantEvaluator(2),
            method='uct',
            epsilon_action=.3,
            tau=1,
        )
        mcts._expand_leaf(root)
        for i, edge in enumerate(root.edges):
            edge.stats.N = i + 1

        pi, values = mcts.get_av()
        np.testing.assert_almost_equal(pi, np.array([1 / 3, 2 / 3, 0.0]), decimal=7)

    @patch('random.uniform', lambda *a, **kw: 0)
    def test_choose_action(self):
        state = self.state
        root = Node(state)
        mcts = MCTS(
            root,
            evaluator=ConstantEvaluator(2),
            method='uct',
        )
        action = mcts.choose_action(pi=np.array([.2, .5, .3]))
        self.assertEqual(1, action)

        mcts.is_training = True
        mcts.epsilon_action = .5
        action = mcts.choose_action(pi=np.array([.33, .34, .33]))
        self.assertEqual(2, action)


class TestMonteCarloRollout(TestCase):
    def setUp(self):
        set_seed()
        self.state = State(first=0, n_pawns=3)

    def test_game(self):
        agent = MonteCarloRolloutAgent(
            mcts_kwargs=dict(
                uct=1,
                method='uct',
                max_steps=10,
            ),
            max_time_per_move=1e9,
        )
        game = Game(agent_0=agent, agent_1='random', n_pawns=3, first=0)
        action_history = game.run()
        self.assertEqual(game.winner, 0)
        self.assertEqual(
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 2, 1, 2, 2, 2, 1, 0, 1, 1, 1, 0, 1],
            action_history
        )

    def test_get_action_tricky(self):
        self.state.set_from_advancement([[0, 4, 8], [5, 2, 8]])
        agent = MonteCarloRolloutAgent(
            mcts_kwargs=dict(
                max_steps=50,
                method='uct',
            ),
            pid=0,
            max_time_per_move=1e9,
        )
        action = agent.get_action(self.state)
        self.assertEqual(0, action)

    def test_mcts_kwargs(self):
        mcts_kwargs = dict(uct=1)
        passed_mcts_kwargs = copy.deepcopy(mcts_kwargs)
        MonteCarloRolloutAgent(mcts_kwargs=passed_mcts_kwargs)
        self.assertEqual(mcts_kwargs, passed_mcts_kwargs)


class TestMonteCarloQLearning(TestCase):
    def setUp(self):
        set_seed()
        self.state = State(first=0, n_pawns=3)

    def test_game(self):
        agent = MonteCarloQLearningAgent(
            mcts_kwargs=dict(
                uct=1,
                method='uct',
                max_steps=10,
            ),
            max_time_per_move=1e9,
        )
        game = Game(agent_0=agent, agent_1='random', n_pawns=3, first=0)
        game.run()

    def test_mcts_kwargs(self):
        mcts_kwargs = dict(uct=1)
        passed_mcts_kwargs = copy.deepcopy(mcts_kwargs)
        MonteCarloQLearningAgent(mcts_kwargs=passed_mcts_kwargs)
        self.assertEqual(mcts_kwargs, passed_mcts_kwargs)


class TestMonteCarloDeepQLearning(TestCase):
    def setUp(self):
        set_seed()
        self.state = State(first=0, n_pawns=3)

    def test_game(self):
        with TemporaryDirectory() as model_path:
            agent = MonteCarloDeepQLearningAgent(
                mcts_kwargs=dict(
                    uct=1,
                    method='uct',
                    max_steps=10,
                ),
                max_time_per_move=1e9,
                model_path=model_path,
            )
            game = Game(agent_0=agent, agent_1='random', n_pawns=3, first=0)
            game.run()

    def test_mcts_kwargs(self):
        mcts_kwargs = dict(uct=1)
        passed_mcts_kwargs = copy.deepcopy(mcts_kwargs)
        MonteCarloDeepQLearningAgent(mcts_kwargs=passed_mcts_kwargs)
        self.assertEqual(mcts_kwargs, passed_mcts_kwargs)
