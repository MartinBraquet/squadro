import numpy as np
from numpy._typing import NDArray

from squadro.state.state import State


def get_reward(winner: int, cur_player: int, return_all: bool) -> NDArray[np.float64]:
    """
    Get the double-value reward for a given winner.
    """
    v1 = 2 * winner - 1
    reward = np.array([-v1, v1], dtype=np.float32)
    if not return_all:
        reward = reward[cur_player]
    return reward


def get_grid_shape(n_pawns: int):
    return [2 * (n_pawns + 1) + 1] * n_pawns * 2 + [2]


def state_to_index(state: State):
    dims = get_grid_shape(state.n_pawns)
    x, y = state.get_advancement()
    return np.ravel_multi_index(x + y + [state.cur_player], dims=dims)


def index_to_state(index: int, n_pawns: int):
    shape = get_grid_shape(n_pawns)
    return np.unravel_index(index, shape=shape)
