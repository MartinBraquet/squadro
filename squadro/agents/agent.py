from abc import ABC, abstractmethod

from squadro.squadro_state import SquadroState


class Agent(ABC):
    """
    Abstract class that represents an agent.
    """

    def __init__(self):
        self.id = None

    def __repr__(self):
        return self.get_name()

    @abstractmethod
    def get_action(self, state: SquadroState, last_action: int, time_left: float):
        """
        Compute the action to perform on the current state
        of the game. The must be computed in at most time_left
        seconds.

        state: the current state
        time_left: the number of second left
        """
        pass

    @classmethod
    def get_name(cls):
        return 'unnamed'

    def set_id(self, _id):
        """
        Set the id of the agent in the game. In a two player
        game it will be either 0 if we play first of 1 otherwise.
        """
        self.id = _id

    def to_dict(self):
        return {
            'name': self.get_name(),
        }
