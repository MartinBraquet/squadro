import random
from copy import deepcopy
from functools import lru_cache

import pygame

from squadro.tools.constants import DefaultParams
from squadro.tools.serialize import hash_dict

# Allowed moves
MOVES = [[1, 3, 2, 3, 1, 3, 2, 1, 2], [3, 1, 2, 1, 3, 1, 2, 3, 2]]
MOVES_RETURN = [[3, 1, 2, 1, 3, 1, 2, 3, 2], [1, 3, 2, 3, 1, 3, 2, 1, 2]]
MAX_PAWNS = len(MOVES[0])


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
    init_pos = [[n_pawns + 1] * n_pawns for _ in range(2)]
    return init_pos


# @lru_cache
# def get_init_return_pos(n_pawns):
#     init_pos = [[0] * n_pawns for _ in range(2)]
#     return init_pos


class State:
    """
    Player 0 is yellow, starting at the bottom
    Player 1 is red, starting on the right
    Coordinates start from the left top corner

    With n, the number of pawns:
    The possible positions are from 0 to n+1 (included)
    Both player's pawns start at n+1
    Since all pawns move along the same horizontal or vertical line, only the other component
    is stored
    For example, the first pawn of P0 moves along Y=1, so we store the X coordinate only.
    Example pos after one move action = 2 from P1: [[4, 4, 4], [4, 4, 3]]
    """

    def __init__(self, n_pawns=None, first=None):
        self.cur_player = 0
        self.winner = None
        self.timeout_player = None
        self.invalid_player = None
        if first is not None:
            assert first in [0, 1], "first must be 0 or 1"
            self.cur_player = first
        else:
            self.cur_player = random.randint(0, 1)
        self.first = self.cur_player
        self.n_pawns = int(n_pawns) if n_pawns is not None else DefaultParams.n_pawns
        assert 2 <= self.n_pawns <= MAX_PAWNS, f"n_pawns must be between 2 and {MAX_PAWNS}"
        # Position of the pawns
        self.pos = get_init_pos(self.n_pawns)
        # Are the pawns on their return journey ?
        self.returning = [[False] * self.n_pawns for _ in range(2)]
        # Have the pawns completed their journey ?
        self.finished = [[False] * self.n_pawns for _ in range(2)]
        self.total_moves = 0

    def __repr__(self):
        # return f'turn: {self.cur_player}, winner: {self.winner}'
        return f'{self.pos}'

    def __eq__(self, other):
        return self.cur_player == other.cur_player and self.pos == other.pos

    def to_dict(self):
        return {
            'cur_player': self.cur_player,
            'pos': self.pos,
            'returning': self.returning,
            'finished': self.finished,
        }

    @property
    def hash(self):
        return hash_dict(self.to_dict())

    @property
    def max_pos(self):
        return self.n_pawns + 1

    @property
    def n_pawns_to_win(self):
        return self.n_pawns - 1

    def set_timed_out(self, player):
        self.timeout_player = player
        self.winner = 1 - player

    def set_invalid_action(self, player, action='unknown'):
        raise ValueError(f'Invalid action for player {player}, action: {action}')

    def get_pawn_position(self, player, pawn):
        """
        Returns the position of the requested pawn ((x, y) position on the board)
        (X, Y) where Y goes downward and X goes to the right, transposed of matrices
        Player 0's moves increase with Y
        Player 1's moves increase with X

            0 ... n+1    X
        0
        ...       ...    P1
        n+1   ... ...

        Y     P0
        """
        if 0 > pawn or pawn > self.n_pawns - 1:
            raise ValueError('pawn must be between 0 and n_pawns - 1')
        if player == 0:
            return pawn + 1, self.pos[player][pawn]
        elif player == 1:
            return self.pos[player][pawn], pawn + 1
        raise ValueError('player must be 0 or 1')

    def get_pawn_advancement(self, player, pawn):
        """
        Returns the number of tiles the pawn has advanced, i.e., {0, ..., 2 * (n_pawns + 1)}
        """
        if self.is_pawn_finished(player, pawn):
            return 2 * self.max_pos
        elif self.is_pawn_returning(player, pawn):
            pos = self.get_pawn_position(player, pawn)[1 - player]
            return self.max_pos + pos
        else:
            pos = self.max_pos - self.get_pawn_position(player, pawn)[1 - player]
            return pos

    def set_from_advancement(self, advancement: list[list[int]]):
        self.returning = [[a >= self.max_pos for a in pa] for pa in advancement]
        self.finished = [[a == 2 * self.max_pos for a in pa] for pa in advancement]
        self.pos = [[abs(a - self.max_pos) for a in pa] for pa in advancement]
        self.game_over_check()

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
        cp = State()
        cp.cur_player = self.cur_player
        cp.winner = self.winner
        cp.timeout_player = self.timeout_player
        cp.invalid_player = self.invalid_player
        cp.pos = deepcopy(self.pos)
        cp.returning = deepcopy(self.returning)
        cp.finished = deepcopy(self.finished)
        cp.n_pawns = self.n_pawns
        cp.first = self.first
        cp.total_moves = self.total_moves
        return cp

    def get_init_args(self):
        return {
            'first': self.first,
            'n_pawns': self.n_pawns,
        }

    def game_over(self):
        """
        Return true if and only if the game is over (game ended, player timed out or made invalid move).
        """
        if self.winner is not None:
            return True
        return self.game_over_check()

    def game_over_check(self):
        """
        Checks if a player succeeded to win the game, i.e. move n_pawns - 1 pawns to the other side and back again.
        """
        for i in range(2):
            if sum(self.finished[i]) >= self.n_pawns_to_win:
                self.winner = i
                return True
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
        actions = [
            a for a in range(self.n_pawns)
            if not self.finished[self.cur_player][a]
        ]
        return actions

    def get_random_action(self):
        actions = self.get_current_player_actions()
        return random.choice(actions)

    def apply_action(self, action):
        """
        Applies a given action to this state. It assumes that the actions is
        valid. This must be checked with is_action_valid.
        """
        if not self.is_action_valid(action):
            self.set_invalid_action(player=self.cur_player, action=action)

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

        self.total_moves += 1
        self.cur_player = 1 - self.cur_player

    def move_1(self, player, pawn):
        """
        Moves the pawn one tile forward in the correct direction
        """
        if not self.returning[player][pawn]:
            self.pos[player][pawn] -= 1
            if self.pos[player][pawn] <= 0:
                self.returning[player][pawn] = True

        else:
            self.pos[player][pawn] += 1
            if self.pos[player][pawn] >= self.max_pos:
                self.finished[player][pawn] = True

    def return_init(self, player: int, pawn: int) -> None:
        """
        Puts the pawn back at the start (or the return start)
        """
        self.pos[player][pawn] = 0 if self.returning[player][pawn] else self.max_pos

    def check_crossings(self, player: int, pawn: int) -> bool:
        """
        Returns whether the pawn crossed one or more opponents and updates the state accordingly
        """
        ended = False
        crossed = False
        opponent = 1 - player

        while not ended:
            opponent_pawn = self.pos[player][pawn] - 1
            if (
                0 <= opponent_pawn <= self.n_pawns - 1
                and self.pos[opponent][opponent_pawn] == pawn + 1
            ):
                crossed = True
                self.move_1(player, pawn)
                self.return_init(opponent, opponent_pawn)
            else:
                ended = True

        return crossed

    def get_winner(self):
        """
        Get the winner of the game. Call only if the game is over.
        """
        return self.winner

    def anim(self, blocking=True):
        # Avoid circular import, might need to refactor
        from squadro.animation.board import Board, handle_events

        board = Board(self.n_pawns, title=f"State Visualization")
        board.turn_draw(self)
        if blocking:
            try:
                while True:
                    handle_events()
            except SystemExit:
                pass
        else:
            pygame.image.save(board.screen, "/tmp/squadro_state.png")
            import subprocess
            subprocess.Popen(["xdg-open", "/tmp/squadro_state.png"])
