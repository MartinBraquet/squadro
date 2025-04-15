"""
MiniMax and AlphaBeta algorithms.
Author: Cyrille Dejemeppe <cyrille.dejemeppe@uclouvain.be>
Copyright (C) 2014, Universite catholique de Louvain

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""
import json
import logging
from collections import defaultdict
from os import mkdir
from os.path import exists

from squadro.squadro_state import State

inf = float("inf")


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

        def save_nodes(s: State):
            if not hasattr(s, 'children'):
                return s.tree_index
            return {
                s.tree_index: [save_nodes(n) for n in s.children]
            }

        nested_nodes = save_nodes(state)
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
        logging.info(f'Node index #{state.tree_index}: {cls.nodes[state.tree_index]}')

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
