from time import time

from squadro import minimax
from squadro.agents.alphabeta_agent import AlphaBetaAgent

"""
Smart agent
"""


class SmartAgentForTraining(AlphaBetaAgent):

  def __init__(self):
      self.current_depth = 0
      self.max_depth = 9
      self.max_time = 0
      self.start_time = 0
      self.total_time = 0

  def get_name(self):
      return 'Group 13'
    
  """
  This is the smart class of an agent to play the Squadro game.
  """
  def get_action(self, state, last_action, time_left):
      self.last_action = last_action
      self.time_left = time_left
      self.max_time = time_left
      self.current_depth = 0
      self.start_time = time()
      if self.total_time == 0:
          self.total_time = time_left
      self.max_time = time_left
      best_move = 1
      # print(time_left)
      
      # Iterative deepening
      while time() - self.start_time < self.max_time and self.current_depth < self.max_depth:
          #print(time() - self.start_time)
          best_move = minimax.search(state, self)
          self.current_depth += 1
      #print("Finish")
      #print(self.current_depth)
      #print("Time elapsed during smart agent play:", time() - self.start_time)
      
      l1 = []
      l2 = []
      for pawn in [0, 1, 2, 3, 4]:
          l1.append(state.get_pawn_advancement(self.id, pawn))
          l2.append(state.get_pawn_advancement(1 - self.id, pawn))
      print('{} {} {}'.format(l1, l2, best_move))
      
      return best_move


  """
  The cutoff function returns true if the alpha-beta/minimax
  search has to stop and false otherwise.
  """
  def cutoff(self, state, depth):
      return depth > self.current_depth or state.game_over_check() or time() - self.start_time > self.max_time

  """
  The evaluate function must return an integer value
  representing the utility function of the board.
  """
  def evaluate(self, state):
      l1 = []
      l2 = []
      for pawn in [0, 1, 2, 3, 4]:
          l1.append(state.get_pawn_advancement(self.id, pawn))
          l2.append(state.get_pawn_advancement(1 - self.id, pawn))
      l1.sort()
      l2.sort()
      return sum(l1[1:]) - sum(l2[1:])
