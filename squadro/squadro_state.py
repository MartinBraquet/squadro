import random
from copy import deepcopy
from functools import lru_cache

from squadro.state import State
from squadro.tools.constants import DefaultParams

# Allowed moves
MOVES = [[1, 3, 2, 3, 1], [3, 1, 2, 1, 3]]
MOVES_RETURN = [[3, 1, 2, 1, 3], [1, 3, 2, 3, 1]]

@lru_cache
def get_moves(n_pawns):
    moves = [m[:n_pawns].copy() for m in MOVES]
    return moves

@lru_cache
def get_moves_return(n_pawns):
    moves = [m[:n_pawns].copy() for m in MOVES_RETURN]
    return moves

# Removing cache for now to avoid issues when modifying in place the same object
# @lru_cache
def get_init_pos(n_pawns):
    init_pos = [
        [((i + 1) * 100, (n_pawns + 1) * 100) for i in range(n_pawns)],
        [((n_pawns + 1) * 100, (i + 1) * 100) for i in range(n_pawns)],
    ]
    return init_pos


# @lru_cache
def get_init_return_pos(n_pawns):
    init_pos = [
        [((i + 1) * 100, 0) for i in range(n_pawns)],
        [(0, (i + 1) * 100) for i in range(n_pawns)],
    ]
    return init_pos


class SquadroState(State):

    def __init__(self, n_pawns=None, first=None):
        super().__init__()
        if first is not None:
            assert first in [0, 1], "first must be 0 or 1"
            self.cur_player = first
        else:
            self.cur_player = random.randint(0, 1)
        self.n_pawns = int(n_pawns) if n_pawns is not None else DefaultParams.n_pawns
        # Position of the pawns
        self.cur_pos = get_init_pos(self.n_pawns)
        # Are the pawns on their return journey ?
        self.returning = [[False] * self.n_pawns for _ in range(2)]
        # Have the pawns completed their journey ?
        self.finished = [[False] * self.n_pawns for _ in range(2)]

    def __eq__(self, other):
        return self.cur_player == other.cur_player and self.cur_pos == other.cur_pos

    def set_timed_out(self, player):
        self.timeout_player = player
        self.winner = 1 - player

    def set_invalid_action(self, player):
        self.invalid_player = player
        self.winner = 1 - player

    def get_pawn_position(self, player, pawn):
        """
        Returns the position of the requested pawn ((x, y) position on the board (i.e. in multiples of 100))
        """
        return self.cur_pos[player][pawn]

    def get_pawn_advancement(self, player, pawn):
        """
        Returns the number of tiles the pawn has advanced (i.e. {0, ..., 12})
        """
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
                nb = (100 * (self.n_pawns + 1) -
                      self.get_pawn_position(player, pawn)[1]) / 100
            else:
                nb = (100 * (self.n_pawns + 1) -
                      self.get_pawn_position(player, pawn)[0]) / 100
            return int(nb)

    def is_pawn_returning(self, player, pawn):
        """
        Returns whether the pawn is on its return journey or not
        """
        return self.returning[player][pawn]

    def is_pawn_finished(self, player, pawn):
        """
        Returns whether the pawn has finished its journey
        """
        return self.finished[player][pawn]

    def copy(self):
        """
        Return a deep copy of this state.
        """
        cp = SquadroState()
        cp.cur_player = self.cur_player
        cp.winner = self.winner
        cp.timeout_player = self.timeout_player
        cp.invalid_player = self.invalid_player
        cp.cur_pos = deepcopy(self.cur_pos)
        cp.returning = deepcopy(self.returning)
        cp.finished = deepcopy(self.finished)
        cp.n_pawns = self.n_pawns
        return cp

    def game_over(self):
        """
        Return true if and only if the game is over (game ended, player timed out or made invalid move).
        """
        if self.winner != None:
            return True
        return self.game_over_check()

    def game_over_check(self):
        """
        Checks if a player succeeded to win the game, i.e. move self.n_pawns - 1 pawns to the other side and back again.
        """
        if sum(self.finished[0]) >= self.n_pawns - 1:
            self.winner = 0
            return True
        elif sum(self.finished[1]) >= self.n_pawns - 1:
            self.winner = 1
            return True
        else:
            return False

    def get_cur_player(self):
        """
            Return the index of the current player.
        """
        return self.cur_player

    def is_action_valid(self, action):
        """
        Checks if a given action is valid.
        """
        actions = self.get_current_player_actions()
        return action in actions

    def get_current_player_actions(self):
        """
        Get all the actions that the current player can perform.
        """
        actions = []
        for i in range(self.n_pawns):
            if not self.finished[self.cur_player][i]:
                actions.append(i)
        return actions

    def apply_action(self, action):
        """
        Applies a given action to this state. It assumes that the actions is
        valid. This must be checked with is_action_valid.
        """
        fun = get_moves_return if self.returning[self.cur_player][action] else get_moves
        n_moves = fun(self.n_pawns)[self.cur_player][action]

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

    def move_1(self, player, pawn):
        """
        Moves the pawn one tile forward in the correct direction
        """
        if player == 0:
            if not self.returning[player][pawn]:
                self.cur_pos[player][pawn] = (self.cur_pos[player][pawn][0],
                                              self.cur_pos[player][pawn][
                                                  1] - 100)
                if self.cur_pos[player][pawn][1] <= 0:
                    self.returning[player][pawn] = True

            else:
                self.cur_pos[player][pawn] = (self.cur_pos[player][pawn][0],
                                              self.cur_pos[player][pawn][
                                                  1] + 100)
                if self.cur_pos[player][pawn][1] >= 100 * (self.n_pawns + 1):
                    self.finished[player][pawn] = True

        else:

            if not self.returning[player][pawn]:
                self.cur_pos[player][pawn] = (
                    self.cur_pos[player][pawn][0] - 100,
                    self.cur_pos[player][pawn][1])
                if self.cur_pos[player][pawn][0] <= 0:
                    self.returning[player][pawn] = True

            else:
                self.cur_pos[player][pawn] = (
                    self.cur_pos[player][pawn][0] + 100,
                    self.cur_pos[player][pawn][1])
                if self.cur_pos[player][pawn][0] >= 100 * (self.n_pawns + 1):
                    self.finished[player][pawn] = True

    def return_init(self, player, pawn):
        """
        Puts the pawn back at the start (or the return start)
        """
        fun = (
            get_init_return_pos
            if self.returning[player][pawn]
            else get_init_pos
        )
        self.cur_pos[player][pawn] = fun(self.n_pawns)[player][pawn]

    def check_crossings(self, player, pawn):
        """
        Returns whether the pawn crossed one or more opponents and updates the state accordingly
        """
        ended = False
        crossed = False

        if player == 0:
            while not ended:
                opponent_pawn = int(self.cur_pos[0][pawn][1] / 100) - 1

                if not 0 <= opponent_pawn <= self.n_pawns - 1:
                    ended = True
                elif self.cur_pos[1][opponent_pawn][0] != self.cur_pos[0][pawn][
                    0]:
                    ended = True

                else:
                    crossed = True
                    self.move_1(0, pawn)
                    self.return_init(1, opponent_pawn)

        else:
            while not ended:
                opponent_pawn = int(self.cur_pos[1][pawn][0] / 100) - 1

                if not 0 <= opponent_pawn <= self.n_pawns - 1:
                    ended = True
                elif (
                    self.cur_pos[0][opponent_pawn][1]
                    != self.cur_pos[1][pawn][1]
                ):
                    ended = True

                else:
                    crossed = True
                    self.move_1(1, pawn)
                    self.return_init(0, opponent_pawn)

        return crossed

    def get_scores(self):
        """
        Return the scores of each players.
        """
        pass

    def get_winner(self):
        """
        Get the winner of the game. Call only if the game is over.
        """
        return self.winner

    def get_state_data(self):
        """
        Return the information about the state that is given to students.
        Usually they have to implement their own state class.
        """
        pass
