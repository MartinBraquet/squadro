import torch
from torch import Tensor

from squadro.core.state import State


def get_num_channels(n_pawns: int, board_flipping: bool, separate_networks) -> int:
    return 6 + 4 * n_pawns - int(board_flipping or separate_networks)


def get_channels(state: State, board_flipping=True, separate_networks=False) -> list[Tensor]:
    """
    Get a list of grid where each grid is a binary tensor of shape (d, d)
    where the value is 1 if the pawn is on that grid and 0 otherwise.
    """
    cur_player = state.cur_player
    d = state.grid_dim
    channels = []

    # positions, one plane per piece
    position_channels = get_position_channels(state, board_flipping)
    channels += position_channels

    # speed, one plane per piece
    speed_channels = get_speed_channels(state, board_flipping)
    channels += speed_channels

    # returning, one plane per player
    returning_channels = get_returning_channels(state, board_flipping)
    channels += returning_channels

    # advancement, one plane per player
    advancement_channels = get_advancement_channels(state, board_flipping)
    channels += advancement_channels

    # current player, one plane
    if not (board_flipping or separate_networks):
        grid = torch.ones((d, d)) * cur_player
        channels.append(grid)

    # turn count, one plane
    grid = torch.ones((d, d)) * state.turn_count / state.max_moves
    channels.append(grid)

    return channels


def get_position_channels(state: State, board_flipping: bool):
    channels = []

    for p_id, player_pos in enumerate(state.pos):
        c = []
        for i, p in enumerate(player_pos):
            grid = torch.zeros((state.grid_dim, state.grid_dim))
            idx = (i + 1, p) if p_id == 1 else (p, i + 1)
            grid[idx] = 1
            c.append(grid)
        channels.append(c)

    channels = flip_piece_boards(channels, state, board_flipping)

    return channels


def get_speed_channels(state: State, board_flipping: bool):
    channels = []

    piece_movements = state.get_piece_movements()
    for p_id, player_pos in enumerate(state.pos):
        c = []
        for i, p in enumerate(player_pos):
            grid = torch.zeros((state.grid_dim, state.grid_dim))
            idx = (i + 1, p) if p_id == 1 else (p, i + 1)
            if state.finished[p_id][i]:
                movement = 0
            else:
                movement = piece_movements[p_id * state.n_pawns + i] / 3.
            grid[idx] = movement
            c.append(grid)
        channels.append(c)

    channels = flip_piece_boards(channels, state, board_flipping)

    return channels


def get_returning_channels(state: State, board_flipping: bool):
    channels = []

    for p_id, player_pos in enumerate(state.pos):
        grid = torch.zeros((state.grid_dim, state.grid_dim))
        for i, p in enumerate(player_pos):
            idx = (i + 1, p) if p_id == 1 else (p, i + 1)
            if state.finished[p_id][i]:
                direction = 0
            else:
                direction = -1 if state.returning[p_id][i] else 1
            grid[idx] = direction
        channels.append(grid)

    channels = flip_boards(channels, state, board_flipping)

    return channels


def get_advancement_channels(state: State, board_flipping: bool):
    channels = []
    max_advancement = 2 * state.max_pos

    for p_id, player_pos in enumerate(state.pos):
        grid = torch.zeros((state.grid_dim, state.grid_dim))
        for i, p in enumerate(player_pos):
            idx = (i + 1, p) if p_id == 1 else (p, i + 1)
            grid[idx] = state.get_pawn_advancement(p_id, i) / max_advancement
        channels.append(grid)

    channels = flip_boards(channels, state, board_flipping)

    return channels


def flip_boards(channels, state, board_flipping):
    if board_flipping and state.cur_player == 1:
        channels = list(reversed(channels))
        channels = [g.T for g in channels]
    return channels


def flip_piece_boards(channels, state, board_flipping):
    if board_flipping and state.cur_player == 1:
        channels = list(reversed(channels))
        for i in range(len(channels)):
            channels[i] = [g.T for g in channels[i]]
    channels = [g for c in channels for g in c]
    return channels
