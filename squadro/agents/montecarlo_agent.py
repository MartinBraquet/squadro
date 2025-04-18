"""
Advantages compared to minimax:
MCTS does not necessarily require a heuristic as it can do random playouts, good when no knowledge of the game
MCTS accuracy increases smoothly with computing time, since every node expansion slightly modifies the decision weights.
Minimax accuracy is more like a step function
due to iterative deepening. It makes no update on the decision weights (i.e., action to take) until it finishes exploring the
tree all the way down to depth k, which takes many node expansions.

Most of the time for MCTS is spent on the random playouts (function to compute next state from random action)
Most of the time for Minimax is spent on state copies, keeping them in memory, and their evaluation
"""
import json
import os
import random
from collections import defaultdict
from time import time
from typing import Optional

import numpy as np

from squadro.agents.agent import Agent
from squadro.state import State, get_next_state
from squadro.tools.constants import DefaultParams, inf
from squadro.tools.evaluation import evaluate_advancement
from squadro.tools.log import monte_carlo_logger as logger
from squadro.tools.probabilities import get_random_sample
from squadro.tools.tree import get_nested_nodes


class Node:
    def __init__(self, state: State, depth: int = None):
        self.state = state
        self.edges = []
        self.depth = depth

    def is_leaf(self):
        return len(self.edges) == 0

    @property
    def player_turn(self):
        return self.state.get_cur_player()

    def __repr__(self):
        return repr(self.state)

    def get_edge_stats(self, to_string=False):
        stats = [edge.stats for edge in self.edges]
        if to_string:
            stats = '\n'.join(str(s) for s in stats)
        return stats

    @property
    def children(self):
        return [edge.out_node for edge in self.edges]


def save_edge_values(d: dict, node: Node):
    for edge in node.edges:
        value = edge.stats.N
        idx = (edge.in_node.tree_index, edge.out_node.tree_index)
        d[idx] = value
        save_edge_values(d, edge.out_node)


class Debug:
    tree_wanted = False
    nodes = defaultdict(dict)
    node_counter = 0

    @classmethod
    def save_tree(cls, node: Node):
        if not cls.tree_wanted:
            return
        if not os.path.exists('results'):
            os.mkdir('results')

        with open('results/edges.json', 'w') as f:
            json.dump(cls.edges, f, indent=4)

        edge_values = {}
        save_edge_values(edge_values, node)
        edge_values = {str(key): value for key, value in edge_values.items()}
        with open('results/edge_values.json', 'w') as f:
            json.dump(edge_values, f, indent=4)

        with open('results/nodes.json', 'w') as f:
            json.dump(cls.nodes, f, indent=4)

        nested_nodes = get_nested_nodes(node)
        with open('results/nested_nodes.json', 'w') as f:
            json.dump(nested_nodes, f, indent=4)

    @classmethod
    def clear(cls, node: Node):
        if not cls.tree_wanted:
            return
        cls.edges = []
        cls.nodes = defaultdict(dict)
        cls.node_counter = 0
        if hasattr(node, 'tree_index'):
            del node.tree_index
        cls.save_node(node)

    @classmethod
    def save_node(cls, node: Node):
        if not cls.tree_wanted:
            return
        if not hasattr(node, 'tree_index'):
            node.tree_index = cls.node_counter
            cls.node_counter += 1
        cls.nodes[node.tree_index] |= {
            # 'eval': eval_type,
            'state': str(node.state),
            # 'value': value,
            'depth': node.depth,
        }
        logger.info(f'Node index #{node.tree_index}: {cls.nodes[node.tree_index]}')

    @classmethod
    def save_edge(cls, parent: Node, child: Node):
        if not cls.tree_wanted:
            return
        if not hasattr(parent, 'tree_index'):
            parent.tree_index = cls.node_counter
            cls.node_counter += 1
        if not hasattr(child, 'tree_index'):
            child.tree_index = cls.node_counter
            cls.node_counter += 1
        cls.edges.append((parent.tree_index, child.tree_index))


class Stats:
    def __init__(self, prior: float):
        self.N = 0
        self.W = .0
        self.P = prior

    def update(self, value: float):
        self.N += 1
        self.W += value
        logger.debug(f'Updating edge with value {value}: {self}')

    @property
    def Q(self) -> float:  # noqa
        if self.N == 0:
            return 0
        return self.W / self.N

    def __repr__(self):
        return f'N={self.N}, W={self.W}, Q={round(self.Q, 3)}, P={self.P}'


class Edge:
    def __init__(
        self,
        in_node: Node,
        out_node: Node,
        prior: float,
        action: int,
    ):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.stats = Stats(prior)

    @property
    def player_turn(self):
        return self.in_node.player_turn

    def __repr__(self):
        return f"{self.action}, {self.in_node}->{self.out_node}"


class MCTS:
    """
    Source:
    https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/MCTS.py
    """

    def __init__(self, root: Node, method: str = None):
        self.root = root
        self.method = method or DefaultParams.mcts_method
        if self.method not in self.get_available_methods():
            raise self._method_error()

        # Upper Confidence Bound for Trees (exploration constant)
        self.uct = DefaultParams.get_uct(n_pawns=self.root.state.n_pawns)

        # Probability to choose randomly the root edge during training
        self.epsilon_mcts = .2

        self.is_training = False

        logger.info(f"Using MCTS method '{self.method}', uct={self.uct}")

    @staticmethod
    def get_available_methods():
        return ['p_uct', 'uct', 'biased_uct']

    def _method_error(self):
        return ValueError(
            f"Unknown method '{self.method}', must be one of {self.get_available_methods()}"
        )

    def _get_sim_edge_values(self, node) -> list[float]:
        # Introduce slight incentives for the root node to explore different edges (for training)
        epsilon = self.epsilon_mcts if node == self.root and self.is_training else 0
        nu = np.random.dirichlet([0.8] * len(node.edges))
        nb = sum(edge.stats.N for edge in node.edges)

        values = []
        for i, edge in enumerate(node.edges):
            if self.method == 'p_uct':
                p_stochastic = (1 - epsilon) * edge.stats.P + epsilon * nu[i]
                u = self.uct * p_stochastic * np.sqrt(nb) / (1 + edge.stats.N)
            elif self.method == 'uct':
                if edge.stats.N == 0:
                    u = inf
                else:
                    # constant typically between 1 and 2
                    u = self.uct * np.sqrt(np.log(nb) / edge.stats.N)
            elif self.method == 'biased_uct':
                u = self.uct * self._get_heuristic(edge) / (1 + edge.stats.N)
            else:
                raise self._method_error()
            qu = edge.stats.Q + u
            values.append(qu)

        return values

    def _get_heuristic(self, edge):
        """
        Heuristic for biased UCT, must have the same bounds as Q.
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

    def move_to_leaf(self):
        logger.debug('Sim step 1: MOVING TO LEAF')
        breadcrumbs = []
        node = self.root

        while not node.is_leaf():
            edge_best = self._pick_sim_edge(node)
            node = edge_best.out_node
            breadcrumbs.append(edge_best)

        return node, breadcrumbs

    @staticmethod
    def back_fill(
        leaf: Node,
        value: float,
        breadcrumbs: list,
    ):
        logger.debug('Sim step 3: DOING BACK-FILL')
        log_breadcrumbs(breadcrumbs)
        for edge in breadcrumbs:
            direction = 1 if edge.player_turn == leaf.player_turn else -1
            edge.stats.update(value * direction)


class MonteCarloAgent(Agent):
    """
    Class that represents an agent performing Monte-Carlo tree search.
    """

    def __init__(self, max_time=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_time = max_time or DefaultParams.max_time_per_move  # use fixed time for now
        self.start_time = None

        self.mcts = None
        self.is_training = False
        self.mc_steps = 10_000  # Number of steps in MCTS
        self.epsilon_move = 0.03  # Probability to sample the move from pi
        self.tau = 1  # If MCTS stochastic: select action with distribution pi^(1/tau)

    @classmethod
    def get_name(cls) -> str:
        return "mcts"

    def get_action(
        self,
        state: State,
        last_action: Optional[int] = None,
        time_left: Optional[float] = None,
    ) -> int:
        self.start_time = time()

        root = Node(state, depth=0)
        self.mcts = MCTS(root)
        Debug.clear(root)

        # Needs at least two simulations to give back-fill value to one root edge
        n = 0
        while (time() - self.start_time < self.max_time or n <= 1) and n < self.mc_steps:
            logger.debug(f'\nSIMULATION {n}')
            self.simulate()
            n += 1

        logger.info(f'\n{n} simulations performed.\n'
                    f'Root edges:\n{root.get_edge_stats(to_string=True)}')

        pi, values = self.get_av()
        action = self.choose_action(pi)

        value = values[action]
        # next_state, _ = self.take_action(state, action)
        # - sign because it evaluates with respect to the current player of the state
        # NN_value = -self.evaluate(next_state)[1]

        logger.info(f'Action probs: {pi}\n'
                    f'Action chosen: {action}\n'
                    f'MCTS perceived value: {value:.4f}')

        Debug.save_tree(root)

        return action

    def simulate(self):
        leaf, breadcrumbs = self.mcts.move_to_leaf()
        value = self.evaluate_leaf(leaf)
        self.mcts.back_fill(leaf, value, breadcrumbs)

    @staticmethod
    def evaluate(state: State):
        """
        Evaluate the current state (Q value), according to the current player.
        """
        p = .2 * np.ones(state.n_pawns)
        value = evaluate_advancement(
            state=state,
            player_id=state.cur_player,
        )
        return p, value

    def _expand_leaf(self, leaf: Node, probs: np.ndarray[float]) -> None:
        if leaf.state.game_over():
            return
        allowed_actions = leaf.state.get_current_player_actions()
        probs = probs[allowed_actions]
        for idx, action in enumerate(allowed_actions):
            state = get_next_state(state=leaf.state, action=action)
            node = Node(state, depth=leaf.depth + 1)

            # if state.id not in self.mcts.tree:
            Debug.save_node(node=node)
            Debug.save_edge(parent=leaf, child=node)
            # logger.info('added node......p = %f', probs[idx])
            # else:
            #    node = self.mcts.tree[state.id]
            #   logger.info('existing node...%s...', node.id)

            edge = Edge(in_node=leaf, out_node=node, prior=probs[idx], action=action)
            leaf.edges.append(edge)

    def evaluate_leaf(self, leaf: Node):
        logger.debug('Sim step 2: EVALUATING LEAF')

        # if stop_sim:
        #     return 1 if leaf.player_turn == self.id else -1

        probs, value = self.evaluate(leaf.state)
        logger.debug(f'Predicted value for player {leaf.player_turn}: {value}')

        self._expand_leaf(leaf, probs)

        return value

    def get_av(self):
        """
        Get Action-Value pairs for each edge of the root node
        """
        n_pawns = self.mcts.root.state.n_pawns
        pi = np.zeros(n_pawns, dtype=np.integer)
        values = np.zeros(n_pawns, dtype=np.float32)

        for edge in self.mcts.root.edges:
            pi[edge.action] = pow(edge.stats.N, 1 / self.tau)
            values[edge.action] = edge.stats.Q
        pi = pi / float(np.sum(pi))
        return pi, values

    def choose_action(self, pi: np.ndarray) -> int:
        """
        Pick the action.
        With probability epsilon (e.g., 3% of picks), draw a random sample from the distribution pi
        (training only)
        Otherwise, pick deterministically the action with the highest probability in pi
        """
        if self.is_training and random.uniform(0, 1) < self.epsilon_move:
            action = get_random_sample(pi)
        else:
            actions = np.argwhere(pi == max(pi))
            # If several actions have the same prob, pick randomly among them
            action = random.choice(actions)[0]
        return int(action)


def log_breadcrumbs(bread):
    if not bread:
        return
    text = 'Breadcrumbs:'
    for edge in bread:
        text += f'\n{edge}'
    logger.debug(text)
