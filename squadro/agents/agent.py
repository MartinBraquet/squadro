from squadro import minimax


class Agent:
    """
    Class that represents an agent.
    """

    def __repr__(self):
        return self.get_name()

    def get_action(self, state, last_action, time_left):
        """
        Compute the action to perform on the current state
        of the game. The must be computed in at most time_left
        seconds.

        state: the current state
        time_left: the number of second left
        """
        pass

    def get_name(self):
        return 'student'

    def set_id(self, id):
        """
        Set the id of the agent in the game. In a two player
        game it will be either 0 if we play first of 1 otherwise.
        """
        self.id = id

    def to_dict(self):
        return {
            'name': self.get_name(),
        }


class AlphaBetaAgent(Agent):
    """
    Alpha beta agent.
    """

    def get_action(self, state, last_action, time_left):
        """This function is used to play a move according
        to the board, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        """
        return minimax.search(state, self)

    def successors(self, state):
        """The successors function must return (or yield) a list of
        pairs (a, s) in which a is the action played to reach the
        state s;"""
        pass

    def cutoff(self, state, depth):
        """The cutoff function returns true if the alpha-beta/minimax
        search has to stop; false otherwise.
        """
        pass

    def evaluate(self, state):
        """The evaluate function must return an integer value
        representing the utility function of the board.
        """
        pass

