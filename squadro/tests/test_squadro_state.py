import pytest

from squadro.squadro_state import SquadroState


@pytest.fixture
def sample_state():
    """Provides a sample state for tests."""
    return SquadroState(n_pawns=5, first=0)


def test_apply_action_valid_action(sample_state):
    """Test apply_action with a valid action."""
    valid_action = 1  # Assuming 1 is a valid action for the initial state
    sample_state.apply_action(valid_action)
    assert sample_state.get_pawn_position(0, valid_action) == (2, 3)  # Position should update
    assert sample_state.pos[0][1] == 3
    assert sample_state.get_pawn_advancement(0, 1) == 3

    sample_state.apply_action(0)  # let other play one move

    assert not sample_state.is_pawn_returning(0, 1)
    sample_state.apply_action(valid_action)
    assert sample_state.is_pawn_returning(0, 1)
    for i in range(6):
        assert not sample_state.is_pawn_finished(0, 1)
        sample_state.apply_action(0)  # let other play one move
        sample_state.apply_action(valid_action)
    assert sample_state.is_pawn_finished(0, 1)

    sample_state.apply_action(0)  # let other play one move
    assert sample_state.get_current_player_actions() == [0, 2, 3, 4]


def test_initial_state_attributes(sample_state):
    """Test the initial state attributes."""
    assert sample_state.cur_player == 0  # Player 0 should be the default first player
    assert sample_state.n_pawns == 5  # Default number of pawns should be 5
    assert sample_state.total_moves == 0  # No moves have been made yet
    assert sample_state.pos == [[6, 6, 6, 6, 6], [6, 6, 6, 6, 6]]  # Default positions
    assert sample_state.returning == [[False] * 5, [False] * 5]  # No pawns are returning yet
    assert sample_state.finished == [[False] * 5, [False] * 5]  # No pawns have finished


def test_apply_action_invalid_action(sample_state):
    """Test apply_action with an invalid action."""
    invalid_action = -1  # Assuming -1 is invalid
    with pytest.raises(ValueError):
        sample_state.apply_action(invalid_action)


def test_check_crossings_no_crossing(sample_state):
    """Test check_crossings when there is no crossing."""
    player = 0
    pawn = 1
    sample_state.pos = [[1, 2, 6, 6, 6], [6, 6, 6, 6, 6]]  # Simulate initial positions
    crossed = sample_state.check_crossings(player, pawn)
    assert not crossed


def test_check_crossings_with_opponent_reset(sample_state):
    """Test check_crossings when there is a crossing and the opponent gets reset."""
    player = 0
    pawn = 4
    sample_state.pos = [[6, 6, 6, 6, 5],
                        [6, 6, 6, 6, 5]]  # Opponent pawn position matches crossing condition
    crossed = sample_state.check_crossings(player, pawn)
    assert crossed
    assert sample_state.pos[1][4] == sample_state.max_pos  # Opponent pawn should reset to max pos


def test_check_crossings_with_multiple_pawns(sample_state):
    """Test check_crossings when multiple opponent pawns could be crossed."""
    player = 0
    pawn = 4
    sample_state.pos = [[6, 6, 6, 6, 5], [6, 6, 6, 5, 5]]  # Multiple pawns aligned for crossing
    crossed = sample_state.check_crossings(player, pawn)
    assert crossed
    assert sample_state.pos[1][4] == sample_state.max_pos  # First opponent pawn reset
    assert sample_state.pos[1][
               3] == sample_state.max_pos  # Second opponent pawn reset as part of chain


def test_apply_action_changes_player(sample_state):
    """Test if apply_action changes the current player."""
    valid_action = 0  # Assuming 0 is valid
    initial_player = sample_state.get_cur_player()
    sample_state.apply_action(valid_action)
    assert sample_state.get_cur_player() != initial_player


def test_apply_action_updates_total_moves(sample_state):
    """Test if the total_moves counter updates after applying an action."""
    initial_moves = sample_state.total_moves
    valid_action = 0  # Assuming 0 is a valid action
    sample_state.apply_action(valid_action)
    assert sample_state.total_moves == initial_moves + 1


def test_return_init_resets_pawn_position(sample_state):
    """Test if return_init correctly resets pawn position."""
    player = 0
    pawn = 1
    sample_state.pos[player][pawn] = 4  # Simulate pawn has moved
    sample_state.returning[player][pawn] = False  # Simulate pawn is not returning
    sample_state.return_init(player, pawn)  # Call return_init
    assert sample_state.pos[player][pawn] == sample_state.max_pos

    # Simulate a returning pawn
    sample_state.returning[player][pawn] = True
    sample_state.return_init(player, pawn)
    assert sample_state.pos[player][pawn] == 0
