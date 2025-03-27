from squadro import minimax
from squadro.agent import AlphaBetaAgent


class MyAgent(AlphaBetaAgent):
    """
    Basic agent
    """


def get_action(self, state, last_action, time_left):
    """
    This is the basic class of an agent to play the Squadro game.
    """
    self.last_action = last_action
    self.time_left = time_left
    return minimax.search(state, self)


def successors(self, state):
    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """
    actions = state.get_current_player_actions()
    for a in actions:
        s = state.copy()
        s.apply_action(a)
        yield (a, s)


def cutoff(self, state, depth):
    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """
    return depth > 0 or state.game_over_check()


def evaluate(self, state):
    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """
    return sum(state.get_pawn_advancement(self.id, pawn) for pawn in [0, 1, 2, 3, 4])
