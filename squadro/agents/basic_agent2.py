from squadro import minimax
from squadro.agents.agent import AlphaBetaAgent

"""
Basic agent 2
"""
class MyAgent(AlphaBetaAgent):

  """
  This is the basic class 2 of an agent to play the Squadro game.
  """
  def get_action(self, state, last_action, time_left):
      self.last_action = last_action
      self.time_left = time_left
      return minimax.search(state, self)

  """
  The successors function must return (or yield) a list of
  pairs (a, s) in which a is the action played to reach the
  state s.
  """
  def successors(self, state):
      actions = state.get_current_player_actions()
      for a in actions:
          s = state.copy()
          s.apply_action(a)
          yield (a, s)

  """
  The cutoff function returns true if the alpha-beta/minimax
  search has to stop and false otherwise.
  """
  def cutoff(self, state, depth):
      return depth > 0 or state.game_over_check()

  """
  The evaluate function must return an integer value
  representing the utility function of the board.
  """
  def evaluate(self, state):
      return sum(state.get_pawn_advancement(self.id, pawn) - state.get_pawn_advancement(1 - self.id, pawn) for pawn in [0, 1, 2, 3, 4])
