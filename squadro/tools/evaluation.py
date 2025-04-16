from squadro.state import State


def evaluate_advancement(state: State, player_id: int) -> int:
    max_value = state.n_pawns_to_win * state.max_pos * 2
    # max_value = inf
    if state.game_over():
        # Needed, otherwise a winning state might be less than a non-winning state.
        # As a winning state might be as low as 1 (when the opponent is one tile from winning)
        return max_value if state.winner == player_id else -max_value

    advancement = state.get_advancement()
    if player_id == 0:
        l1, l2 = advancement
    else:
        l2, l1 = advancement
    return (sum(l1) - min(l1)) - (sum(l2) - min(l2))
