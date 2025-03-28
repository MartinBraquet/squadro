import json
import signal
from pathlib import Path
from time import time

from squadro.agent import Agent
from squadro.squadro_state import SquadroState
from squadro.tools.constants import DefaultParams
from squadro.tools.utils import get_agent


class GameFromState:
    def __init__(
        self,
        state: SquadroState,
        agent_0: Agent | str = None,
        agent_1: Agent | str = None,
        time_out=None,
    ):
        agent_0 = agent_0 or DefaultParams.agent
        agent_1 = agent_1 or DefaultParams.agent

        self.agents = [get_agent(a) for a in (agent_0, agent_1)]
        self.agents[0].set_id(0)
        self.agents[1].set_id(1)
        self.times_left = [time_out] * 2

        self.state = state

        self.action_history = []

    def __repr__(self):
        text = f'{self.agents[0]} vs {self.agents[1]}, first: {self.first}, {self.state.n_pawns} pawns'
        if self.winner is not None:
            text += f', winner: {self.winner}, {len(self.action_history)} moves'
        return text

    def __eq__(self, other: 'GameFromState'):
        return self.to_dict() == other.to_dict()

    @property
    def title(self):
        return f'{self.agents[0]} vs {self.agents[1]}'

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

    @property
    def n_pawns(self):
        return self.state.n_pawns

    def run(self):
        """
        Run the game if not yet played
        Return the action history
        """
        if not self.action_history:
            last_action = None
            while not self.state.game_over():
                player = self.state.get_cur_player()
                try:
                    action, exe_time = get_timed_action(
                        player=self.agents[player],
                        state=self.state.copy(),
                        last_action=last_action,
                        time_left=self.times_left[player],
                    )
                except TimeoutError:
                    self.state.set_timed_out(player)
                    break

                if self.times_left[player]:
                    self.times_left[player] -= exe_time

                self.action_history.append(action)
                self.state.apply_action(action)
                last_action = action
                self._post_apply_action()

        return self.action_history.copy()

    def _post_apply_action(self):
        pass

    def to_dict(self):
        return {
            'winner': self.winner,
            'agent_0': self.agents[0].get_name(),
            'agent_1': self.agents[1].get_name(),
            'action_history': self.action_history,
            'state': self.state.get_init_args(),
        }

    def save_results(self, filename: str | Path):
        """
        Save the game results on disk

        Note: maybe serialize and save whole objects
        """
        self.run()
        results = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    @classmethod
    def from_file(cls, filename: str | Path):
        with open(filename, 'r') as f:
            results = json.load(f)
        game = GameFromState(
            state=SquadroState(**results['state']),
            agent_0=results['agent_0'],
            agent_1=results['agent_1'],
        )
        game.action_history = results['action_history']
        game.state.winner = results['winner']
        return game


class Game(GameFromState):
    def __init__(
        self,
        n_pawns=None,
        first=None,
        **kwargs,
    ):
        n_pawns = int(n_pawns) if n_pawns is not None else DefaultParams.n_pawns
        state = SquadroState(n_pawns=n_pawns, first=first)
        super().__init__(state=state, **kwargs)


def handle_timeout(signum, frame):
    """
    Define behavior in case of timeout.
    """
    raise TimeoutError


def get_timed_action(player, state, last_action, time_left):
    """
    Get an action from player with a timeout.
    """
    if not time_left:
        return player.get_action(state, last_action, time_left), 0

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, time_left)
    start_time = time()
    try:
        action = player.get_action(state, last_action, time_left)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        exe_time = time() - start_time
    return action, exe_time
