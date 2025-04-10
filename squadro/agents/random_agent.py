import random

from squadro.agents.agent import Agent


class RandomAgent(Agent):

    def get_action(self, state, last_action=None, time_left=None):
        actions = state.get_current_player_actions()
        action = random.choice(actions)
        return action

    @classmethod
    def get_name(cls):
        return "random"
