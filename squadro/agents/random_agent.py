from squadro.agents.agent import Agent


class RandomAgent(Agent):

    def get_action(self, state, last_action=None, time_left=None):
        action = state.get_random_action()
        return action

    @classmethod
    def get_name(cls):
        return "random"
