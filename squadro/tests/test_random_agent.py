from unittest.mock import MagicMock

import pytest

from squadro.random_agent import MyAgent


def test_get_action_returns_valid_action():
    state = MagicMock()
    state.get_current_player_actions.return_value = ["action1", "action2", "action3"]
    agent = MyAgent()

    action = agent.get_action(state, None, 10)

    assert action in state.get_current_player_actions()


def test_get_action_uses_random_choice():
    state = MagicMock()
    state.get_current_player_actions.return_value = ["action1", "action2", "action3"]
    agent = MyAgent()

    actions_taken = set(agent.get_action(state, None, 10) for _ in range(100))

    assert all(action in state.get_current_player_actions() for action in actions_taken)


def test_get_name_returns_random():
    agent = MyAgent()
    assert agent.get_name() == "random"


@pytest.fixture
def mock_agent():
    return MyAgent()


def test_agent_inherits_from_base_class(mock_agent):
    from squadro.agent import Agent
    assert isinstance(mock_agent, Agent)


def test_get_action_no_available_actions():
    state = MagicMock()
    state.get_current_player_actions.return_value = []
    agent = MyAgent()

    with pytest.raises(IndexError):
        agent.get_action(state, None, 10)
