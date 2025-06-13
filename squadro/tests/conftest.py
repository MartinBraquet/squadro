import pytest

from squadro.core.state import State


@pytest.fixture
def sample_state():
    """Provides a sample state for tests."""
    return State(n_pawns=5, first=0)
