from abc import ABC, abstractmethod
from typing import Optional

from squadro.state import State


class Agent(ABC):
    """
    Abstract class that represents an agent.
    """

    def __init__(self, pid=None, max_time_per_move=None):
        self.id = pid
        self.max_time_per_move = max_time_per_move

    def __repr__(self):
        return self.get_name()

    @abstractmethod
    def get_action(
        self,
        state: State,
        last_action: Optional[int] = None,
        time_left: Optional[float] = None,
    ):
        """
        Compute the action to perform on the current state
        of the game. This must be computed in at most time_left
        seconds.

        state: the current state
        time_left: the number of seconds left
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        pass

    def set_id(self, _id):
        """
        Set the id of the agent in the game. In a two-player
        game, it will be either 0 if we play first or 1 otherwise.
        """
        self.id = _id

    def to_dict(self):
        return {
            'name': self.get_name(),
        }
