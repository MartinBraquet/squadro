"""
Advantages compared to minimax:
MCTS does not necessarily require a heuristic as it can do random playouts, good when no knowledge of the game
MCTS accuracy increases smoothly with computing time, since every node expansion slightly modifies the decision weights.
Minimax accuracy is more like a step function
due to iterative deepening. It makes no update on the decision weights (i.e., action to take) until it finishes exploring the
tree all the way down to depth k, which takes many node expansions.

Most of the time for MCTS is spent on the random playouts (function to compute the next state from random action)
Most of the time for Minimax is spent on state copies, keeping them in memory, and their evaluation
"""
import random
from abc import ABC
from time import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from squadro.agents.agent import Agent
from squadro.evaluators.advancement import AdvancementEvaluator
from squadro.evaluators.evaluator import Evaluator
from squadro.evaluators.rl import QLearningEvaluator, DeepQLearningEvaluator
from squadro.evaluators.rollout import RolloutEvaluator
from squadro.state import State, get_next_state
from squadro.tools.constants import DefaultParams, inf
from squadro.tools.evaluation import evaluate_advancement
from squadro.tools.log import monte_carlo_logger as logger
from squadro.tools.probabilities import get_random_index
from squadro.tools.tree import Node, Debug, Edge, log_trajectory


class MCTS:
    """
    Source:
    https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/MCTS.py
    """

    def __init__(
        self,
        root: Node,
        evaluator: Evaluator,
        method: str = None,
        max_time: float = None,
        max_steps: int = None,
        epsilon_action: float = None,
        stochastic_moves: float = None,
        p_mix: float = None,
        a_dirichlet: float = None,
        tau: float = None,
        uct: float = None,
        is_training: bool = None,
    ):
        self.root = root

        self.method = method or DefaultParams.mcts_method
        if self.method not in self.get_available_methods():
            raise self._unknown_method_error()

        self.evaluator = evaluator

        self.max_time = max_time or 1e9
        assert self.max_time > 0

        self.is_training = is_training if is_training is not None else False

        # Max number of steps
        self.max_steps = max_steps or DefaultParams.max_mcts_steps
        assert self.max_steps > 0

        # Probability to sample the action from pi instead of arg max pi during training
        self.epsilon_action = epsilon_action
        # assert 0 <= self.epsilon_action <= 1

        # Choose stochastic actions for the first `stochastic_moves` moves
        if stochastic_moves is not None:
            self.stochastic_moves = stochastic_moves
        else:
            self.stochastic_moves = root.state.max_moves
        assert 0 <= self.stochastic_moves

        # If MCTS stochastic: select action with distribution pi^(1/tau)
        self.tau = tau or 1.
        assert self.tau > 0

        self.decay_rate = .9
        self.min_temp = .1

        # Upper Confidence Bound for Trees (exploration constant)
        self.uct = uct or DefaultParams.get_uct(n_pawns=self.n_pawns)
        assert self.uct > 0

        # Mixing constant between prior and dirichlet noise in P-UCT during training
        self.p_mix = p_mix if p_mix is not None else .4
        assert 0 <= self.p_mix <= 1

        # Dirichlet noise spread.
        # Small value means spiky distribution (one action dominates)
        self.a_dirichlet = a_dirichlet or .9 / self.n_pawns
        assert self.a_dirichlet > 0

        self.move_probs = None

        logger.info(f"Using MCTS method '{self.method}', uct={self.uct}")

    def __repr__(self) -> str:
        return (
            f"{self.evaluator.__class__.__name__}"
            f", method: {self.method}"
            f", uct: {self.uct}"
            f", max_steps: {self.max_steps}"
            f", tau: {self.tau}"
            # f", epsilon_action: {self.epsilon_action}"
            f", stochastic_moves: {self.stochastic_moves}"
        )

    @property
    def config(self) -> dict:
        data = {
            'method': self.method,
            'uct': self.uct,
            'max_steps': self.max_steps,
            'tau': self.tau,
            'min_temp': self.min_temp,
            'decay_rate': self.decay_rate,
            'stochastic_moves': self.stochastic_moves,
            'p_mix': self.p_mix,
            'a_dirichlet': self.a_dirichlet,
            'evaluator': self.evaluator.__class__.__name__,
        }
        if self.epsilon_action is not None:
            data['epsilon_action'] = self.epsilon_action
        return data

    @property
    def n_pawns(self) -> int:
        return self.root.state.n_pawns

    @staticmethod
    def get_available_methods() -> list[str]:
        return ['uct', 'p_uct', 'biased_uct']

    def _unknown_method_error(self) -> ValueError:
        return ValueError(
            f"Unknown method '{self.method}', must be one of {self.get_available_methods()}"
        )

    def _get_sim_edge_values(self, node: Node) -> list[float]:
        # Introduce slight incentives for the root node to explore different edges (for training)
        epsilon = self.p_mix if node == self.root and self.is_training else 0
        nu = np.random.dirichlet([self.a_dirichlet] * len(node.edges))
        nb = sum(edge.stats.N for edge in node.edges)

        values = []
        for i, edge in enumerate(node.edges):
            if self.method == 'uct':
                if edge.stats.N == 0:
                    u = inf
                else:
                    # constant typically between 1 and 2
                    u = self.uct * np.sqrt(np.log(nb) / edge.stats.N) + epsilon * nu[i]
            elif self.method == 'p_uct':
                if edge.stats.P is None:
                    raise ValueError("Cannot use the 'p_uct' method when the prior is not provided")
                p_stochastic = (1 - epsilon) * edge.stats.P + epsilon * nu[i]
                u = self.uct * p_stochastic * np.sqrt(nb) / (1 + edge.stats.N)
            elif self.method == 'biased_uct':
                u = self.uct * self._get_heuristic(edge) / (1 + edge.stats.N)
            else:
                raise self._unknown_method_error()
            qu = edge.stats.Q + u
            values.append(qu)

        return values

    @staticmethod
    def _get_heuristic(edge: Edge) -> float:
        """
        Heuristic for biased UCT (must have the same bounds as Q).

        Args:
            edge: The edge to evaluate

        Returns:
            float: The heuristic value for the edge
        """
        value = evaluate_advancement(
            state=edge.out_node.state,
            player_id=edge.in_node.player_turn,
        )
        return value

    def _pick_sim_edge(self, node: Node) -> Edge:
        qu_all = self._get_sim_edge_values(node)
        i = np.argmax(qu_all)
        edge_best = node.edges[i]
        return edge_best

    def move_to_leaf(self) -> tuple[Node, list[Edge]]:
        logger.debug('Sim step 1: MOVING TO LEAF')
        trajectory = []
        node = self.root

        while not node.is_leaf():
            edge_best = self._pick_sim_edge(node)
            node = edge_best.out_node
            trajectory.append(edge_best)

        return node, trajectory

    def _expand_leaf(self, leaf: Node, probs: NDArray[np.float64] = None) -> None:
        if leaf.state.game_over():
            return
        allowed_actions = leaf.state.get_current_player_actions()
        if probs is not None:
            probs = probs[allowed_actions]
        for idx, action in enumerate(allowed_actions):
            prior = probs[idx] if probs is not None else None
            edge = self.get_edge(
                parent=leaf,
                action=action,
                prior=prior,  # noqa
            )
            # TODO: consider handling transposition table here for faster convergence and less memory usage
            #  But possibly harder to implement backpropagation during reinforcement learning
            # if state.id not in self.mcts.tree:
            # logger.info('added node......p = %f', probs[idx])
            # else:
            #    node = self.mcts.tree[state.id]
            #   logger.info('existing node...%s...', node.id)
            leaf.edges.append(edge)

    @staticmethod
    def get_edge(parent: Node, action: int, prior: float | np.float64 = None) -> Edge:
        state = get_next_state(state=parent.state, action=action)
        child = Node(state, depth=parent.depth + 1)
        edge = Edge(in_node=parent, out_node=child, prior=prior, action=action)
        Debug.save_node(node=child)
        Debug.save_edge(parent=parent, child=child)
        return edge

    def evaluate_leaf(self, leaf: Node) -> float:
        logger.debug('Sim step 2: EVALUATING LEAF')
        probs, value = self.evaluator.evaluate(leaf.state)
        logger.debug(f'Predicted value for player {leaf.player_turn}: {value}')
        self._expand_leaf(leaf, probs=probs)
        return value

    @staticmethod
    def back_fill(
        player_turn: int,
        value: float,
        trajectory: list[Edge],
    ) -> None:
        logger.debug('Sim step 3: DOING BACKFILL')
        log_trajectory(trajectory)
        for edge in trajectory:
            direction = 1 if edge.player_turn == player_turn else -1
            edge.stats.update(value * direction)

    def simulate(self) -> None:
        leaf, trajectory = self.move_to_leaf()
        value = self.evaluate_leaf(leaf)
        self.back_fill(value=value, trajectory=trajectory, player_turn=leaf.player_turn)

    def get_action(self):
        Debug.clear(self.root)
        start_time = time()
        n = 0

        # Needs at least two simulations to give backfill value to one root edge
        while (time() - start_time < self.max_time or n <= 1) and n < self.max_steps:
            logger.debug(f'\nSIMULATION {n}')
            self.simulate()
            n += 1

        logger.info(f'\n{n} simulations performed.\n'
                    f'Root edges:\n{self.root.get_edge_stats(to_string=True)}')

        self.move_probs, values = self.get_av()
        action = self.choose_action(self.move_probs)

        value = values[action]
        # next_state, _ = self.take_action(state, action)
        # - sign because it evaluates with respect to the current player of the state
        # NN_value = -self.evaluate(next_state)[1]

        logger.info(f'Action probs: {np.round(self.move_probs, 3)}\n'
                    f'Action chosen: {action}\n'
                    f'MCTS perceived value: {value:.3f}')

        Debug.save_tree(self.root)

        return action

    def get_av(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get Action-Value pairs for each edge of the root node.

        Returns:
            A tuple containing:
                - np.ndarray: Probability distribution over actions (pi)
                - np.ndarray: Q-values for each action
        """
        n_pawns = self.root.state.n_pawns
        pi = - np.ones(n_pawns, dtype=np.float32) * inf
        values = np.zeros(n_pawns, dtype=np.float32)

        temperature = self.get_temperature()

        for edge in self.root.edges:
            pi[edge.action] = np.log(edge.stats.N + 1e-9) / temperature
            values[edge.action] = edge.stats.Q

        # Numerical trick to avoid overflow when computing softmax (N^temp)
        pi = np.exp(pi - pi.max())
        pi /= pi.sum()

        return pi, values

    def get_temperature(self) -> float:
        """
        Get temperature for the softmax function.
        """
        return max(self.tau * self.decay_rate ** self.root.state.turn_count, self.min_temp)

    def choose_action(self, pi: np.ndarray) -> int:
        """
        Pick the action.
        With probability epsilon (e.g., 30% of picks), draw a random sample from the distribution pi
        (training only)
        Otherwise, deterministically pick the action with the highest probability in pi.

        Args:
            pi: Probability distribution over actions

        Returns:
            int: The chosen action index
        """
        condition = None
        if self.is_training:
            if self.epsilon_action is not None:
                condition = random.uniform(0, 1) < self.epsilon_action
            else:
                condition = self.root.state.turn_count < self.stochastic_moves
        if self.is_training and condition:
            action = get_random_index(pi)
        else:
            actions = np.argwhere(pi == pi.max())
            # If several actions have the same prob, pick randomly among them
            action = random.choice(actions)[0]
        return int(action)


class _MonteCarloAgent(Agent, ABC):
    """
    Class that represents an agent performing Monte-Carlo tree search.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        is_training: bool = False,
        mcts_kwargs: Optional[dict] = None,
        **kwargs
    ):
        # use fixed time for now
        kwargs.setdefault('max_time_per_move', DefaultParams.max_time_per_move)
        super().__init__(**kwargs)
        self.mcts_move_probs = None
        self.mcts_kwargs = mcts_kwargs or {}
        self.mcts_kwargs['is_training'] = is_training
        self.mcts_kwargs['evaluator'] = evaluator

    @property
    def evaluator(self) -> Evaluator:
        return self.mcts_kwargs['evaluator']

    @property
    def max_steps(self) -> int:
        return self.mcts_kwargs['max_steps']

    @property
    def mcts_config(self) -> dict:
        mcts = self.get_mcts(Node(State()))
        return mcts.config

    def get_mcts(self, root):
        mcts = MCTS(
            root=root,
            max_time=self.max_time_per_move,
            **self.mcts_kwargs,
        )
        return mcts

    def get_action(
        self,
        state: State,
        last_action: Optional[int] = None,
        time_left: Optional[float] = None,
    ) -> int:
        root = Node(state)
        mcts = self.get_mcts(root)
        # print(mcts)
        action = mcts.get_action()
        self.mcts_move_probs = mcts.move_probs
        return action

    def get_move_info(self):
        return dict(
            mcts_move_probs=self.mcts_move_probs,
        )


class MonteCarloAdvancementAgent(_MonteCarloAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('evaluator', AdvancementEvaluator())
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "mcts_advancement"


class MonteCarloRolloutAgent(_MonteCarloAgent):
    def __init__(
        self,
        mcts_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        mcts_kwargs = (mcts_kwargs or {}).copy()
        mcts_kwargs.setdefault('method', 'uct')
        kwargs.setdefault('evaluator', RolloutEvaluator())
        super().__init__(mcts_kwargs=mcts_kwargs, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "mcts_rollout"


class MonteCarloQLearningAgent(_MonteCarloAgent):
    def __init__(
        self,
        model_path: str = None,
        mcts_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        mcts_kwargs = (mcts_kwargs or {}).copy()
        mcts_kwargs.setdefault('method', 'uct')
        if 'evaluator' not in kwargs:
            kwargs['evaluator'] = QLearningEvaluator(model_path=model_path)
        super().__init__(mcts_kwargs=mcts_kwargs, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "mcts_q_learning"


class MonteCarloDeepQLearningAgent(_MonteCarloAgent):
    def __init__(
        self,
        model_path: str = None,
        model_config=None,
        mcts_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        mcts_kwargs = (mcts_kwargs or {}).copy()
        mcts_kwargs.setdefault('method', 'p_uct')
        if 'evaluator' not in kwargs:
            kwargs['evaluator'] = DeepQLearningEvaluator(
                model_path=model_path,
                model_config=model_config
            )
        super().__init__(mcts_kwargs=mcts_kwargs, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "mcts_deep_q_learning"
