import logging
from abc import abstractmethod
from time import time

from squadro import minimax
from squadro.agents.agent import Agent
from squadro.minimax import inf
from squadro.squadro_state import State


class AlphaBetaAgent(Agent):
    """
    Abstract class that represents an alpha-beta agent.

    Minimax tree search: assuming that the heuristic evaluation function is perfect, known by both players,
      and that the other player plays perfectly to minimize that state value (worst-case approach),
      MMTS gives the action that maximizes the state value.
    Alpha-beta pruning: modification of MMTS to discard the exploration of nodes that are sure to
      not lead to the highest state value. It only makes the algorithm more efficient, without
      modifying the outcome.
    """

    def get_action(self, state: State, last_action: int = None, time_left: float = None):
        """This function is used to play a move according
        to the board, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        """
        return minimax.search(state, self)

    def successors(self, state: State):
        """The successors function must return (or yield) a list of
        pairs (a, s) in which `a` is the action played to reach the
        state `s`;"""
        actions = state.get_current_player_actions()
        for a in actions:
            s = state.copy()
            s.apply_action(a)
            yield a, s

    @abstractmethod
    def cutoff(self, state: State, depth: int):
        """The cutoff function returns true if the alpha-beta/minimax
        search has to stop; false otherwise.
        """
        pass

    @abstractmethod
    def evaluate(self, state: State):
        """The evaluate function must return a number
        representing the utility function of the board, according to the player doing the minimax
        search (NOT the player for the passed `state` in `state.cur_player`).
        """
        pass


class AlphaBetaAdvancementAgent(AlphaBetaAgent):
    """
    Alpha-beta advancement agent:

    Pick the action according to minimax tree search (with alpha-beta pruning, depth=0),
    where the state evaluation function is the player's advancement.

    A depth of 0 means that it explores all the possible actions, and directly assess them based
    on the evaluation of the states that those actions lead to.
    Player's advancement: number of steps all the pawns have traveled so far.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = 0

    @classmethod
    def get_name(cls):
        return 'ab_advancement'

    def cutoff(self, state: State, depth: int):
        return depth > self.depth or state.game_over_check()

    def evaluate(self, state: State):
        return sum(
            state.get_pawn_advancement(self.id, pawn)
            for pawn in range(state.n_pawns)
        )


class AlphaBetaRelativeAdvancementAgent(AlphaBetaAdvancementAgent):
    """
    Alpha-beta relative-advancement agent:

    Pick the action according to minimax tree search (with alpha-beta pruning, depth=0),
    where the state
    evaluation function is the player's advancement compared to the opponent's advancement.
    """

    @classmethod
    def get_name(cls):
        return 'ab_relative_advancement'

    def evaluate(self, state: State):
        return sum(
            state.get_pawn_advancement(self.id, p) - state.get_pawn_advancement(1 - self.id, p)
            for p in range(state.n_pawns)
        )


class AlphaBetaAdvancementDeepAgent(AlphaBetaAdvancementAgent):
    """
    Alpha-beta deep advancement agent:

    Pick the action according to minimax tree search (with alpha-beta pruning, depth up to 9),
    where the heuristic state
    evaluation function is the player's advancement compared to the opponent's advancement (limited
    to the `n_pawns - 1` most advanced pawns, required to win).

    Also limits the tree search according to the total remaining player's time, via iterative
    deepening. It finds the best move after stopping at depth 1, then finds the best move after
    stopping at depth 2, etc. until time runs out, and it returns the move found at the last depth
    (best move so far).
    """

    def __init__(self, max_depth=9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.max_time = .005  # use fixed time for now
        self.start_time = None
        # self.total_time = None

    @classmethod
    def get_name(cls):
        return 'ab_advancement_deep'

    def get_action(self, state, last_action=None, time_left=None):
        # if time_left is None:
        #     self.depth = self.max_depth
        #     return minimax.search(state, self)

        self.depth = 0
        self.start_time = time()

        # if self.total_time is None:
        #     self.total_time = time_left
        # if time_left / self.total_time > 0.2:
        #     self.max_time = 0.03 * self.total_time
        # else:
        #     self.max_time = 0.03 * self.total_time * (time_left / (0.2 * self.total_time)) ** 2

        # Iterative deepening
        best_move = state.get_random_action()
        while time() - self.start_time < self.max_time and self.depth < self.max_depth:
            minimax_action = minimax.search(state, self)
            if minimax_action is None:
                raise ValueError('No best move found, check cutoff function')
            if time() - self.start_time < self.max_time:
                # Only keep the minimax action computed for the deepest depth if it got time to
                # explore all the leaf nodes at that depth
                best_move = minimax_action
                minimax.Debug.save_tree(state)
            self.depth += 1
            # print(time() - self.start_time)
            # print('depth', self.depth)

        logging.info(f'depth: {self.depth}')
        return best_move

    @property
    def time_is_limited(self):
        return self.max_time

    def cutoff(self, state, depth):
        # Should not cut off at zero depth, otherwise search will not compute
        # the best action, returning ac = None
        return (
            super().cutoff(state, depth)
            or self.time_is_limited and time() - self.start_time > self.max_time and depth > 0
        )

    def evaluate(self, state):
        if state.game_over():
            # Needed, otherwise a winning state might be less than a non-winning state.
            # As a winning state might be as low as 1 (when the opponent is one tile from winning)
            return inf if state.winner == self.id else -inf
        l1 = []
        l2 = []
        for pawn in range(state.n_pawns):
            l1.append(state.get_pawn_advancement(self.id, pawn))
            l2.append(state.get_pawn_advancement(1 - self.id, pawn))
        return (sum(l1) - min(l1)) - (sum(l2) - min(l2))
