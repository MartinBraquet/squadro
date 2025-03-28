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
from squadro.squadro_state import SquadroState

inf = float("inf")


def search(st: SquadroState, player, prune=True):
    """Perform a MiniMax/AlphaBeta search and return the best action.

    Arguments:
    state -- initial state
    player -- a concrete instance of class AlphaBetaPlayer
    prune -- whether to use AlphaBeta pruning

    """

    def max_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            value = player.evaluate(state)
            # print(f'max eval, {value=} {state=}, {depth=}')
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
            # print(f'min eval, {value=} {state=}, {depth=}')
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

    _, ac = max_value(st, -inf, inf, 0)
    if ac is None:
        max_value(st, -inf, inf, 0)
    return ac
