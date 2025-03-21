import minimax

"""
Class that represents a agent.
"""
class Agent():

    """
    Compute the action to perfom on the current state
    of the game. The must be compute in at most time_left
    seconds.

    state: the current state
    time_left: the number of second left

    """
    def get_action(self, state, last_action, time_left):
        abstract

    def get_name(self):
        return 'student agent'

    """
    Set the id of the agent in the game. In a two player 
    game it will be either 0 if we play first of 1 otherwise.
    """
    def set_id(self, id):
        self.id = id

"""
Alpha beta agent.
"""
class AlphaBetaAgent(Agent):

    """This is the skeleton of an agent."""
    
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
        abstract

    def cutoff(self, state, depth):
        """The cutoff function returns true if the alpha-beta/minimax
        search has to stop; false otherwise.
        """
        abstract

    def evaluate(self, state):
        """The evaluate function must return an integer value
        representing the utility function of the board.
        """
        abstract

