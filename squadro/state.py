import random
from copy import deepcopy
from functools import lru_cache
from typing import Tuple

import pygame
import torch

from squadro.tools.constants import DefaultParams
from squadro.tools.serialize import hash_dict

# Allowed moves
MOVES = [[1, 3, 2, 3, 1, 3, 2, 1, 2], [3, 1, 2, 1, 3, 1, 2, 3, 2]]
MOVES_RETURN = [[3, 1, 2, 1, 3, 1, 2, 3, 2], [1, 3, 2, 3, 1, 3, 2, 1, 2]]
MAX_PAWNS = len(MOVES[0])


@lru_cache
def get_moves(n_pawns: int) -> list[list[int]]:
    moves = [m[:n_pawns].copy() for m in MOVES]
    return moves


@lru_cache
def get_moves_return(n_pawns: int) -> list[list[int]]:
    moves = [m[:n_pawns].copy() for m in MOVES_RETURN]
    return moves


def get_piece_movements(advancement: list[int | list] | torch.Tensor) -> list[int]:
    """
    >>> get_piece_movements([0, 4, 3, 1, 5, 8])
    [1, 1, 2, 3, 3, 2]
    >>> get_piece_movements([[0, 4, 3], [1, 5, 8]])
    [1, 1, 2, 3, 3, 2]
    """
    if len(advancement) == 2:
        n_pawns = len(advancement[0])
    else:
        n_pawns = len(advancement) // 2
        advancement = [advancement[:n_pawns], advancement[n_pawns:]]
    max_pos = n_pawns + 1
    moves = get_moves(n_pawns)
    moves_return = get_moves_return(n_pawns)
    steps = [
        (moves_return if m >= max_pos else moves)[p_id][i]
        for p_id, a in enumerate(advancement)
        for i, m in enumerate(a)
    ]
    return steps


# Removing cache for now to avoid issues when modifying in place the same object
# @lru_cache
def get_init_pos(n_pawns: int) -> list[list[int]]:
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
    Both players' pawns start at n+1
    Since all pawns move along the same horizontal or vertical line, only the other component
    is stored.
    For example, the first pawn of P0 moves along Y=1, so we store the X coordinate only.
    Example pos after one move action = 2 from P1: [[4, 4, 4], [4, 4, 3]]
    """

    def __init__(
        self,
        n_pawns: int = None,
        first: int = None,
        advancement: list[list[int]] = None,
        cur_player: int = None,
    ):
        self.winner = None
        self.timeout_player = None
        self.invalid_player = None

        is_init = advancement is None

        if not is_init:
            n_pawns = len(advancement[0])
            if first is None:
                first = 'unknown'

        if first is not None:
            assert first in [0, 1, 'unknown'], f"first must be 0 or 1, not {first}"
            self.first = first
        else:
            self.first = random.randint(0, 1)

        if is_init:
            assert cur_player is None, "cur_player must be None for initial state. Set first instead."
            self.cur_player = self.first
        else:
            self.cur_player = cur_player if cur_player is not None else 0

        self.n_pawns = int(n_pawns) if n_pawns is not None else DefaultParams.n_pawns
        assert 2 <= self.n_pawns <= MAX_PAWNS, f"n_pawns must be between 2 and {MAX_PAWNS}"

        # Position of the pawns
        self.pos = get_init_pos(self.n_pawns)
        self.returning = [[False] * self.n_pawns for _ in range(2)]
        self.finished = [[False] * self.n_pawns for _ in range(2)]
        self.total_moves = 0

        if advancement:
            self.set_from_advancement(advancement)

    def __repr__(self):
        # return f'turn: {self.cur_player}, winner: {self.winner}'
        return f'{self.get_advancement()}'

    def __eq__(self, other: 'State') -> bool:
        return self.cur_player == other.cur_player and self.pos == other.pos

    def to_dict(self) -> dict:
        return {
            'cur_player': self.cur_player,
            'pos': self.pos,
            'returning': self.returning,
            'finished': self.finished,
        }

    @property
    def hash(self) -> int:
        return hash_dict(self.to_dict())

    @property
    def max_pos(self) -> int:
        return self.n_pawns + 1

    @property
    def grid_dim(self) -> int:
        return self.n_pawns + 2

    @property
    def n_pawns_to_win(self) -> int:
        return self.n_pawns - 1

    def set_timed_out(self, player: int) -> None:
        self.timeout_player = player
        self.winner = 1 - player

    def set_invalid_action(self, player: int, action: int | str = 'unknown') -> None:
        raise ValueError(f'Invalid action for player {player}, action: {action}')

    def get_pawn_position(self, player: int, pawn: int) -> Tuple[int, int]:
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

    def get_pawn_advancement(self, player: int, pawn: int) -> int:
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

    def get_advancement(self) -> list[list[int]]:
        """
        Returns the advancement of each player's pawns, in the same format as `self.pos`
        """
        advancement = [
            [
                self.get_pawn_advancement(player_id, pawn)
                for pawn in range(self.n_pawns)
            ]
            for player_id in range(2)
        ]
        return advancement

    def set_from_advancement(self, advancement: list[list[int]]):
        self.returning = [[a >= self.max_pos for a in pa] for pa in advancement]
        self.finished = [[a == 2 * self.max_pos for a in pa] for pa in advancement]
        self.pos = [[abs(a - self.max_pos) for a in pa] for pa in advancement]
        self.game_over_check()

    def is_pawn_returning(self, player: int, pawn: int) -> bool:
        """
        Returns whether the pawn is on its return journey or not
        """
        return self.returning[player][pawn]

    def is_pawn_finished(self, player: int, pawn: int) -> bool:
        """
        Returns whether the pawn has finished its journey
        """
        return self.finished[player][pawn]

    def copy(self) -> 'State':
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

    def get_init_args(self) -> dict:
        return {
            'first': self.first,
            'n_pawns': self.n_pawns,
        }

    def game_over(self) -> bool:
        """
        Return true if and only if the game is over (game ended, player timed out or made an invalid move).
        """
        if self.winner is not None:
            return True
        return self.game_over_check()

    def game_over_check(self) -> bool:
        """
        Checks if a player succeeded to win the game, i.e., move n_pawns - 1 pawns to the other side and back again.
        """
        for i in range(2):
            if sum(self.finished[i]) >= self.n_pawns_to_win:
                self.winner = i
                return True
        return False

    def get_cur_player(self) -> int:
        """
        Return the index of the current player.
        """
        return self.cur_player

    def is_action_valid(self, action: int) -> bool:
        """
        Checks if a given action is valid.
        """
        actions = self.get_current_player_actions()
        return action in actions

    def get_current_player_actions(self) -> list[int]:
        """
        Get all the actions that the current player can perform.
        """
        actions = [
            a for a in range(self.n_pawns)
            if not self.finished[self.cur_player][a]
        ]
        return actions

    def get_random_action(self) -> int:
        actions = self.get_current_player_actions()
        return random.choice(actions)

    def get_next_state(self, action: int) -> 'State':
        return get_next_state(self, action=action)

    def apply_action(self, action: int) -> None:
        """
        Applies a given action to this state. It assumes that the action is
        valid. This must be checked with is_action_valid.
        Note that it modifies the object in place.
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

    def move_1(self, player: int, pawn: int) -> None:
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

    def get_winner(self) -> int:
        """
        Get the winner of the game. Call only if the game is over.
        """
        return self.winner

    def anim(self, blocking: bool = True) -> None:
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

    def to_list(self):
        """
        List representation of the state, used for the replay buffer or other serialized things.
        """
        x = [self.get_advancement(), self.cur_player]
        return x

    @classmethod
    def from_list(cls, arr: list):
        """
        Create a state from a list representation.

        >>> State.from_list([[[1, 2, 3], [1, 2, 4]], 0]).to_list()
        [[[1, 2, 3], [1, 2, 4]], 0]
        """
        advancement, cur_player = arr
        n_pawns = len(advancement[0])
        state = cls(n_pawns=n_pawns, advancement=advancement, cur_player=cur_player)
        return state

    def get_piece_movements(self):
        return get_piece_movements(self.get_advancement())


def get_next_state(state: State, action: int) -> State:
    new_state = state.copy()
    new_state.apply_action(action)
    return new_state
