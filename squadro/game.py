from squadro.squadro_state import SquadroState
from squadro.tools.constants import DefaultParams
from squadro.tools.utils import get_agent


class Game:
    def __init__(
        self,
        agent_0=None,
        agent_1=None,
        first=None,
        n_pawns=None,
    ):
        agent_0 = agent_0 if agent_0 is not None else DefaultParams.agent
        agent_1 = agent_1 if agent_1 is not None else DefaultParams.agent
        n_pawns = int(n_pawns) if n_pawns is not None else DefaultParams.n_pawns

        self.agents = [get_agent(a) for a in (agent_0, agent_1)]
        self.agents[0].set_id(0)
        self.agents[1].set_id(1)

        self.state = SquadroState(n_pawns=n_pawns, first=first)

        self.action_history = []

    def __repr__(self):
        text = f'{self.agents[0]} vs {self.agents[1]}, first: {self.first}, {self.state.n_pawns} pawns'
        if self.winner is not None:
            text += f', winner: {self.winner}, {len(self.action_history)} moves'
        return text

    @property
    def first(self):
        """
        Index of the first player
        """
        return self.state.first

    @property
    def winner(self):
        """
        Index of the winner
        """
        return self.state.winner

    def run(self):
        last_action = None
        while not self.state.game_over():
            player = self.state.get_cur_player()
            action = self.agents[player].get_action(
                state=self.state.copy(),
                last_action=last_action,
                time_left=50,
            )
            # print('player', player, 'action', action)
            self.action_history.append(action)

            if self.state.is_action_valid(action):
                self.state.apply_action(action)
                last_action = action
            else:
                self.state.set_invalid_action(player)

        return self.action_history
