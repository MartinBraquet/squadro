import minimax
import random
from state import State
from copy import deepcopy

# Pawn initial positions
INIT_POS = [[(100, 600), (200, 600), (300, 600), (400, 600), (500, 600)], [(600, 100), (600, 200), (600, 300), (600, 400), (600, 500)]]
INIT_RET_POS = [[(100, 0), (200, 0), (300, 0), (400, 0), (500, 0)], [(0, 100), (0, 200), (0, 300), (0, 400), (0, 500)]]

# Allowed moves
MOVES = [[1, 3, 2, 3, 1], [3, 1, 2, 1, 3]]
MOVES_RETURN = [[3, 1, 2, 1, 3], [1, 3, 2, 3, 1]]

class SquadroState(State):

  def __init__(self):
    self.cur_player = random.randint(0, 1)
    self.winner = None
    self.timeout_player = None
    self.invalid_player = None
    # Position of the pawns
    self.cur_pos = deepcopy(INIT_POS)
    # Are the pawns on their return journey ?
    self.returning = [[False, False, False, False, False], [False, False, False, False, False]]
    # Have the pawns completed their journey ?
    self.finished = [[False, False, False, False, False], [False, False, False, False, False]]


  def __eq__(self, other):
    return self.cur_player == other.cur_player and self.cur_pos == other.cur_pos

  
  def set_timed_out(self, player):
    self.timeout_player = player
    self.winner = 1 - player
  

  def set_invalid_action(self, player):
    self.invalid_player = player
    self.winner = 1 - player


  """
  Returns the position of the requested pawn ((x, y) position on the board (i.e. in multiples of 100))
  """
  def get_pawn_position(self, player, pawn):
    return self.cur_pos[player][pawn]


  """
  Returns the number of tiles the pawn has advanced (i.e. {0, ..., 12})
  """
  def get_pawn_advancement(self, player, pawn):
    if self.is_pawn_finished(player, pawn):
      return 12
    elif self.is_pawn_returning(player, pawn):
      if player == 0:
        nb = self.get_pawn_position(player, pawn)[1] / 100
      else:
        nb = self.get_pawn_position(player, pawn)[0] / 100
      return int(6 + nb)
    else:
      if player == 0:
        nb = (600 - self.get_pawn_position(player, pawn)[1]) / 100
      else:
        nb = (600 - self.get_pawn_position(player, pawn)[0]) / 100
      return int(nb)


  """
  Returns whether the pawn is on its return journey or not
  """
  def is_pawn_returning(self, player, pawn):
    return self.returning[player][pawn]


  """
  Returns whether the pawn has finished its journey
  """
  def is_pawn_finished(self, player, pawn):
    return self.finished[player][pawn]


  """
  Return a deep copy of this state.
  """
  def copy(self):
    cp = SquadroState()
    cp.cur_player = self.cur_player
    cp.winner = self.winner
    cp.timeout_player = self.timeout_player
    cp.invalid_player = self.invalid_player
    cp.cur_pos = deepcopy(self.cur_pos)
    cp.returning = deepcopy(self.returning)
    cp.finished = deepcopy(self.finished)
    return cp


  """
  Return true if and only if the game is over (game ended, player timed out or made invalid move).
  """
  def game_over(self):
    if self.winner != None:
      return True
    return self.game_over_check()
  

  """
  Checks if a player succeeded to win the game, i.e. move 4 pawns to the other side and back again.
  """
  def game_over_check(self):
    if sum(self.finished[0]) >= 4:
      self.winner = 0
      return True
    elif sum(self.finished[1]) >= 4:
      self.winner = 1
      return True
    else:
      return False


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
    actions = []
    for i in range(5):
      if not self.finished[self.cur_player][i]:
        actions.append(i)
    return actions



  """
  Applies a given action to this state. It assume that the actions is
  valid. This must be checked with is_action_valid.
  """
  def apply_action(self, action):
    if not self.returning[self.cur_player][action]:
      n_moves = MOVES[self.cur_player][action]
    else:
      n_moves = MOVES_RETURN[self.cur_player][action]
    
    for i in range(n_moves):
      ret_before = self.returning[self.cur_player][action]
      self.move_1(self.cur_player, action)
      
      if self.returning[self.cur_player][action] != ret_before:
        break
      if self.finished[self.cur_player][action]:
        break
      if self.check_crossings(self.cur_player, action):
        break
    
    self.cur_player = 1 - self.cur_player


  """
  Moves the pawn one tile forward in the correct direction
  """
  def move_1(self, player, pawn):
    if player == 0:
      if not self.returning[player][pawn]:
        self.cur_pos[player][pawn] = (self.cur_pos[player][pawn][0], self.cur_pos[player][pawn][1] - 100)
        if (self.cur_pos[player][pawn][1] <= 0):
          self.returning[player][pawn] = True
      
      else:
        self.cur_pos[player][pawn] = (self.cur_pos[player][pawn][0], self.cur_pos[player][pawn][1] + 100)
        if (self.cur_pos[player][pawn][1] >= 600):
          self.finished[player][pawn] = True
    
    else:
      
      if not self.returning[player][pawn]:
        self.cur_pos[player][pawn] = (self.cur_pos[player][pawn][0] - 100, self.cur_pos[player][pawn][1])
        if (self.cur_pos[player][pawn][0] <= 0):
          self.returning[player][pawn] = True
      
      else:
        self.cur_pos[player][pawn] = (self.cur_pos[player][pawn][0] + 100, self.cur_pos[player][pawn][1])
        if (self.cur_pos[player][pawn][0] >= 600):
          self.finished[player][pawn] = True


  """
  Puts the pawn back at the start (or the return start)
  """
  def return_init(self, player, pawn):
    if not self.returning[player][pawn]:
      self.cur_pos[player][pawn] = INIT_POS[player][pawn]
    else:
      self.cur_pos[player][pawn] = INIT_RET_POS[player][pawn]


  """
  Returns whether the pawn crossed one or more opponents and updates the state accordingly
  """
  def check_crossings(self, player, pawn):
    ended = False
    crossed = False
    
    if player == 0:
      while not ended:
        opponent_pawn = int(self.cur_pos[0][pawn][1] / 100) - 1
        
        if not 0 <= opponent_pawn <= 4:
          ended = True
        elif self.cur_pos[1][opponent_pawn][0] != self.cur_pos[0][pawn][0]:
          ended = True
        
        else:
          crossed = True
          self.move_1(0, pawn)
          self.return_init(1, opponent_pawn)

    else:
      while not ended:
        opponent_pawn = int(self.cur_pos[1][pawn][0] / 100) - 1
        
        if not 0 <= opponent_pawn <= 4:
          ended = True
        elif self.cur_pos[0][opponent_pawn][1] != self.cur_pos[1][pawn][1]:
          ended = True
        
        else:
          crossed = True
          self.move_1(1, pawn)
          self.return_init(0, opponent_pawn)

    return crossed
    

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
