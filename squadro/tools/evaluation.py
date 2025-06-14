from squadro.state.state import State


def evaluate_advancement(state: State, player_id: int = None) -> int:
    """
    Evaluate the current state (Q value), according to the player in player id.
    Normalized between -1 and 1.

    :param state: State to evaluate
    :param player_id: Player to evaluate (default: current player)
    :return: Q value
    """
    max_value = state.n_pawns_to_win * state.max_pos * 2

    if player_id is None:
        player_id = state.cur_player

    def get_summed_advancement():
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

    value = get_summed_advancement()
    value /= max_value
    return value
