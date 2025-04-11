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

from squadro.squadro_state import SquadroState

inf = float("inf")


class Debug:
    SAVE_TREE = False
    TREE = []


def search(st: SquadroState, player, prune=True):
    """Perform a MiniMax/AlphaBeta search and return the best action.

    Arguments:
    state -- initial state
    player -- a concrete instance of class AlphaBetaPlayer
    prune -- whether to use AlphaBeta pruning

    """

    def max_value(state, alpha, beta, depth):
        # Should not enter this clause at the first iteration, otherwise search will not compute
        # the best action, returning ac = None
        if player.cutoff(state, depth) and depth > 0:
            value = player.evaluate(state)
            if Debug.SAVE_TREE:
                logging.info(f'max eval: {value=}, {state=}, {depth=}')
                Debug.TREE.append({
                    'eval': 'max',
                    'state': str(state.pos),
                    'value': value,
                    'depth': depth,
                })
            return value, None
        val = -inf
        action = None
        for a, s in player.successors(state):
            v, _ = min_value(s, alpha, beta, depth + 1)
            if v > val:
                val = v
                action = a
                if prune:
                    if v >= beta:
                        return v, a
                    alpha = max(alpha, v)
        return val, action

    def min_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            value = player.evaluate(state)
            if Debug.SAVE_TREE:
                logging.info(f'min eval: {value=}, {state=}, {depth=}')
                Debug.TREE.append({
                    'eval': 'min',
                    'state': str(state.pos),
                    'value': value,
                    'depth': depth,
                })
            return value, None
        val = inf
        action = None
        for a, s in player.successors(state):
            v, _ = max_value(s, alpha, beta, depth + 1)
            if v < val:
                val = v
                action = a
                if prune:
                    if v <= alpha:
                        return v, a
                    beta = min(beta, v)
        return val, action

    logging.info(f'Current state: {st}')
    if Debug.SAVE_TREE:
        Debug.TREE = []
    _, ac = max_value(st, -inf, inf, 0)
    if ac is None:
        # max_value(st, -inf, inf, 0)
        if Debug.SAVE_TREE:
            json.dump(Debug.TREE, open('tree.json', 'w'), indent=4)
    return ac
