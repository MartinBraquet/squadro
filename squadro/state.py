import minimax

"""
State history
"""

class State():

  def __init__(self):
    self.cur_player = 0
    self.winner = None
    self.timeout_player = None
    self.invalid_player = None
  
  def set_timed_out(self, player):
    self.timeout_player = player
    self.winner = 1 - player
  
  def set_invalid_action(self, player):
    self.invalid_player = player
    self.winner = 1 - player

  """
  Return a deep copy of this state.
  """
  def copy(self):
    pass

  """
  Return true if and only if the game is over.
  """
  def game_over(self):
    if self.winner != None:
      return True
    self.game_over_check()
  
  def game_over_check(self):
    pass

  """
  Return the index of the current player.
  """
  def get_cur_player(self):
    return self.cur_player

  """
  Checks if a given action is valid.
  """
  def is_action_valid(self, action):
    actions = self.get_current_player_actions()
    return action in actions

  """
  Get all the actions that the current player can perform.
  """
  def get_current_player_actions(self):
    pass

  """
  Applies a given action to this state. It assume that the actions is
  valid. This must be checked with is_action_valid.
  """
  def apply_action(self, action):
    pass

  """
  Return the scores of each players.
  """
  def get_scores(self):
    pass

  """
  Get the winner of the game. Call only if the game is over.
  """
  def get_winner(self):
    return self.winner

  """
  Return the information about the state that is given to students.
  Usually they have to implement their own state class.
  """
  def get_state_data(self):
    pass
