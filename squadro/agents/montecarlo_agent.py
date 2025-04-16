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
import random
from time import time
from typing import Optional

import numpy as np

from squadro.agents.agent import Agent
from squadro.state import State
from squadro.tools.constants import inf
from squadro.tools.evaluation import evaluate_advancement
from squadro.tools.probabilities import get_random_sample


class Node:
    def __init__(self, state: State):
        self.state = state
        self.edges = []

    def is_leaf(self):
        return len(self.edges) == 0

    @property
    def player_turn(self):
        return self.state.get_cur_player()

    def __repr__(self):
        return repr(self.state)


class Stats:
    def __init__(self, prior: float):
        self.N = 0
        self.W = .0
        self.P = prior

    def update(self, value: float):
        self.N += 1
        self.W += value

    @property
    def Q(self) -> float:  # noqa
        if self.N == 0:
            return 0
        return self.W / self.N

    def __repr__(self):
        return f'N={self.N}, W={self.W}, Q={self.Q}, P={self.P}'


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

    def __init__(self, root: Node):
        self.root = root
        self.cpuct = 0.1

    def move_to_leaf(self, player):
        # logger_mcts.info('------MOVING TO LEAF------')

        breadcrumbs = []
        node = self.root

        done = False

        while not node.is_leaf():
            # logger_mcts.info('PLAYER TURN...%d', node.playerTurn)

            qu_max = -inf

            # Choose randomly at 20% for the root node (ONLY for training)
            epsilon_mcts = .2
            epsilon = epsilon_mcts if node == self.root else 0

            # TODO: keep only first line of nu, as it should only be used when node == self.root
            nu = (
                np.random.dirichlet([0.8] * len(node.edges))
                if node == self.root else
                [0] * len(node.edges)
            )

            nb = sum(edge.stats.N for edge in node.edges)

            edge_best = None
            for i, edge in enumerate(node.edges):
                u = (
                    self.cpuct * ((1 - epsilon) * edge.stats.P + epsilon * nu[i])
                    * np.sqrt(nb) / (1 + edge.stats.N)
                )
                # u = self.cpuct * ((1-epsilon) + epsilon * nu[i]) * np.sqrt(nb) / (1 + edge.stats.N)
                # u = self.cpuct * np.sqrt(nb) / (1 + edge.stats.N)

                # logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, q = %f, u = %f, q+u = %f'
                #    , action, action % 7, edge.stats.N, np.round(edge.stats.P,6), np.round(nu[i],6), ((1-epsilon) * edge.stats.P + epsilon * nu[i] )
                #    , np.round(edge.stats['W'],6), np.round(q,6), np.round(u,6), np.round(q+u,6))

                qu = edge.stats.Q + u
                if qu > qu_max:
                    qu_max = qu
                    edge_best = edge

            # logger_mcts.info('action with highest q + u...%d', sim_action)

            # the value of the new_state from the POV of the new playerTurn
            node = edge_best.out_node
            breadcrumbs.append(edge_best)

            done = node.state.game_over()
            if done:
                break

        # logger_mcts.info('DONE...%d', done)
        return node, done, breadcrumbs

    @staticmethod
    def back_fill(
        leaf: Node,
        value: float,
        breadcrumbs: list,
    ):
        # logger_mcts.info('------DOING BACKFILL------')
        # print_breadcrumbs(breadcrumbs)
        for edge in breadcrumbs:
            direction = 1 if edge.player_turn == leaf.player_turn else -1
            edge.stats.update(value * direction)
            # logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
            #    , value * direction
            #    , playerTurn
            #   , edge.stats.N
            #  , edge.stats['W']
            #  , edge.stats.Q
            #  )
            # render(edge.outNode.state, logger_mcts)


class MonteCarloAgent(Agent):
    """
    Class that represents an agent performing Monte-Carlo tree search.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_time = 1e9  # use fixed time for now
        self.start_time = None

        self.mcts = None
        self.mc_steps = 50  # Number of steps in MCTS
        self.epsilon_move = 0.03  # Probability to choose randomly the move
        self.epsilon_mcts = 0.2  # Probability to choose randomly the node in MCTS (for TRAINING only)
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
        # action = state.get_random_action()

        root = Node(state)
        self.mcts = MCTS(root)

        # root_value = self.evaluateLeaf(root)
        n = 0
        while time() - self.start_time < self.max_time and n < self.mc_steps:
            # print(time() - self.start_time)
            # logger_mcts.info('***************************')
            # logger_mcts.info('****** SIMULATION %d ******', n)
            # logger_mcts.info('***************************')
            self.simulate()
            n += 1
        # print("Finish")
        # print(self.current_depth)
        # print("Time elapsed during smart agent play:", time() - self.start_time)

        pi, values = self.get_av()

        action = self.choose_action(pi)

        # value = values[action]
        # next_state, _ = self.take_action(state, action)
        # - sign because it evaluates with respect to the current player of the state
        # NN_value = -self.evaluate(next_state)[1]

        # logger_mcts.info('ACTION VALUES...%s', pi)
        # logger_mcts.info('CHOSEN ACTION...%d', action)
        # logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)  # Value estimated by MCTS: Q = W/N (average of the all the values along the path)
        # logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value) # Value estimated by the Neural Network (only for the next state)

        # print(action, pi, value, NN_value)

        return action

    def simulate(self):
        # logger_mcts.info('ROOT NODE...')
        # render(self.mcts.root.state, logger_mcts)
        # logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.playerTurn)

        leaf, done, breadcrumbs = self.mcts.move_to_leaf(self)
        # render(leaf.state, logger_mcts)
        # print_state(leaf.state)

        value = self.evaluate_leaf(leaf, done)
        self.mcts.back_fill(leaf, value, breadcrumbs)

    @staticmethod
    def evaluate(state: State):
        """
        Evaluate the current state, according to the current player
        """
        p = .2 * np.ones(state.n_pawns)
        value = evaluate_advancement(
            state=state,
            player_id=state.cur_player,
        )
        return p, value

    def evaluate_leaf(self, leaf: Node, done=False):
        # logger_mcts.info('------EVALUATING LEAF------')

        if not done:
            probs, value = self.evaluate(leaf.state)
            # print(probs)
            allowed_actions = leaf.state.get_current_player_actions()
            # logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.playerTurn, value)
            probs = probs[allowed_actions]

            # Add node in tree
            for idx, action in enumerate(allowed_actions):
                state, _ = self.take_action(state=leaf.state, action=action)
                node = Node(state)
                # if state.id not in self.mcts.tree:
                # self.mcts.addNode(node)
                # logger_mcts.info('added node......p = %f', probs[idx])
                # else:
                #    node = self.mcts.tree[state.id]
                #   logger_mcts.info('existing node...%s...', node.id)

                edge = Edge(in_node=leaf, out_node=node, prior=probs[idx], action=action)
                leaf.edges.append(edge)

        else:  # End of game leaf
            # value = 1 if leaf.player_turn == self.id else -1
            _, value = self.evaluate(leaf.state)

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
        With probability epsilon (e.g., 3% of picks), draw a sample from the distribution pi
        Otherwise, pick the action with the highest probability in pi
        """
        if random.uniform(0, 1) < self.epsilon_move:  # Choose stochastically (for TRAINING)
            action = get_random_sample(pi)
        else:  # Choose deterministically
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]  # if several states have the same prob
        return int(action)

    def take_action(self, state: State, action: int) -> (State, int):
        new_state = state.copy()
        new_state.apply_action(action)

        done = 1 if self.cutoff(new_state) else 0

        return new_state, done

    def cutoff(self, state: State):
        return False


##############################################################################

### SET all #logger_DISABLED to True to disable logging
### WARNING: the mcts log file gets big quite quickly

logger_DISABLED = {
    'main': False
    , 'memory': False
    , 'tourney': False
    , 'mcts': False
    , 'model': False}


# logger_mcts = setup_logger('logger_mcts', 'logs/logger_mcts.log')
# logger_mcts.disabled = logger_DISABLED['mcts']

def render(state, logger):
    logger.info(state.pos)
    logger.info('--------------')


def print_breadcrumbs(bread):
    print('Breadcrumbs:')
    for edge in bread:
        print(edge.in_node.state.get_advancement())
        print('=>')
        print(edge.out_node.state.get_advancement())
        print('------------------------------------')
