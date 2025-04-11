from unittest.mock import patch

import pytest

from squadro.agents.random_agent import RandomAgent
from squadro.squadro_state import SquadroState


@patch.object(SquadroState, 'get_current_player_actions',
              lambda self: ["action1", "action2", "action3"])
def test_get_action_returns_valid_action():
    state = SquadroState()
    agent = RandomAgent()

    action = agent.get_action(state)

    assert action in state.get_current_player_actions()


@patch.object(SquadroState, 'get_current_player_actions',
              lambda self: ["action1", "action2", "action3"])
def test_get_action_uses_random_choice():
    state = SquadroState()
    agent = RandomAgent()

    actions_taken = set(agent.get_action(state) for _ in range(100))

    assert all(action in state.get_current_player_actions() for action in actions_taken)


def test_get_name_returns_random():
    agent = RandomAgent()
    assert agent.get_name() == "random"


@pytest.fixture
def mock_agent():
    return RandomAgent()


def test_agent_inherits_from_base_class(mock_agent):
    from squadro.agents.agent import Agent
    assert isinstance(mock_agent, Agent)


@patch.object(SquadroState, 'get_current_player_actions', lambda self: [])
def test_get_action_no_available_actions():
    state = SquadroState()
    agent = RandomAgent()

    with pytest.raises(IndexError):
        agent.get_action(state)


def test_to_dict():
    agent = RandomAgent()
    assert agent.to_dict() == {'name': 'random'}
