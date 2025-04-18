import json
from collections import defaultdict
from os import mkdir
from os.path import exists

from squadro.state import State
from squadro.tools.constants import inf
from squadro.tools.log import alpha_beta_logger as logger
from squadro.tools.tree import get_nested_nodes


class Debug:
    tree_wanted = False
    edges = []
    nodes = defaultdict(dict)
    node_counter = 0

    @classmethod
    def save_tree(cls, state: State):
        if not cls.tree_wanted:
            return
        if not exists('results'):
            mkdir('results')
        with open('results/edges.json', 'w') as f:
            json.dump(cls.edges, f, indent=4)
        with open('results/nodes.json', 'w') as f:
            json.dump(cls.nodes, f, indent=4)

        nested_nodes = get_nested_nodes(state)
        with open('results/nested_nodes.json', 'w') as f:
            json.dump(nested_nodes, f, indent=4)

    @classmethod
    def clear(cls, state: State):
        if not cls.tree_wanted:
            return
        cls.edges = []
        cls.nodes = defaultdict(dict)
        cls.node_counter = 0
        if hasattr(state, 'tree_index'):
            del state.tree_index
        if hasattr(state, 'children'):
            del state.children

    @classmethod
    def save_node(cls, value, state: State, eval_type, depth):
        if not cls.tree_wanted:
            return
        if not hasattr(state, 'tree_index'):
            state.tree_index = cls.node_counter
            cls.node_counter += 1
        cls.nodes[state.tree_index] |= {
            'eval': eval_type,
            'state': str(state.pos),
            'value': value,
            'depth': depth,
        }
        logger.info(f'Node index #{state.tree_index}: {cls.nodes[state.tree_index]}')

    @classmethod
    def save_edge(cls, parent: State, child: State):
        if not cls.tree_wanted:
            return
        if not hasattr(parent, 'tree_index'):
            parent.tree_index = cls.node_counter
            cls.node_counter += 1
        if not hasattr(child, 'tree_index'):
            child.tree_index = cls.node_counter
            cls.node_counter += 1
        if not hasattr(parent, 'children'):
            parent.children = []
        parent.children.append(child)
        cls.edges.append((parent.tree_index, child.tree_index))


def search(st: State, player, prune=True):
    """Perform a MiniMax/AlphaBeta search and return the best action.

    Arguments:
    state -- initial state
    player -- a concrete instance of class AlphaBetaPlayer
    prune -- whether to use AlphaBeta pruning

    """

    def max_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            value = player.evaluate(state)
            Debug.save_node(value, state, 'max', depth)
            return value, None
        value = -inf
        action = None
        for a, s in player.successors(state):
            Debug.save_edge(state, s)
            v, _ = min_value(s, alpha, beta, depth + 1)
            if v > value or v == -inf and action is None:
                value = v
                action = a
                if prune:
                    if v >= beta:
                        break
                    alpha = max(alpha, v)
        Debug.save_node(value, state, 'max', depth)
        return value, action

    def min_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            value = player.evaluate(state)
            Debug.save_node(value, state, 'min', depth)
            return value, None
        value = inf
        action = None
        for a, s in player.successors(state):
            Debug.save_edge(state, s)
            v, _ = max_value(s, alpha, beta, depth + 1)
            if v < value or v == inf and action is None:
                value = v
                action = a
                if prune:
                    if v <= alpha:
                        break
                    beta = min(beta, v)
        Debug.save_node(value, state, 'min', depth)
        return value, action

    Debug.clear(st)
    _, ac = max_value(st, -inf, inf, 0)
    # Debug.save_tree(st)
    # if ac is None:
    #   max_value(st, -inf, inf, 0)
    return ac
