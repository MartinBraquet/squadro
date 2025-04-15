class State:
    """
    State history
    """

    def __init__(self):
        self.cur_player = 0
        self.winner = None
        self.timeout_player = None
        self.invalid_player = None

    def set_timed_out(self, player):
        self.timeout_player = player
        self.winner = 1 - player

    def set_invalid_action(self, player):
        self.invalid_player = player
        self.winner = 1 - player

    def copy(self):
        """
        Return a deep copy of this state.
        """
        pass

    def game_over(self):
        """
        Return true if and only if the game is over.
        """
        if self.winner is not None:
            return True
        self.game_over_check()

    def game_over_check(self):
        pass

    def get_cur_player(self):
        """
        Return the index of the current player.
        """
        return self.cur_player

    def is_action_valid(self, action):
        """
        Checks if a given action is valid.
        """
        actions = self.get_current_player_actions()
        return action in actions

    def get_current_player_actions(self):
        """
        Get all the actions that the current player can perform.
        """
        pass

    def apply_action(self, action):
        """
        Applies a given action to this state. It assumes that the actions is
        valid. This must be checked with is_action_valid.
        """
        pass

    def get_winner(self):
        """
        Get the winner of the game. Call only if the game is over.
        """
        return self.winner
