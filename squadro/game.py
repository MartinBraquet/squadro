import json
import signal
from pathlib import Path
from time import time

from squadro.agents.agent import Agent
from squadro.state import State
from squadro.tools.agents import get_agent
from squadro.tools.arrays import box, array2string
from squadro.tools.constants import DefaultParams
from squadro.tools.log import game_logger as logger


class GameFromState:
    def __init__(
        self,
        state: State,
        agent_0: Agent | str = None,
        agent_1: Agent | str = None,
        time_out=None,
        save_states: bool = None,
        agent_kwargs: dict = None,
        evaluators=None,
    ):
        agent_0 = agent_0 or DefaultParams.agent
        agent_1 = agent_1 or DefaultParams.agent
        agent_kwargs = agent_kwargs or {}

        self.agents = [get_agent(a, pid=i, **agent_kwargs) for i, a in
                       enumerate((agent_0, agent_1))]
        self.times_left = [time_out] * 2

        self.state = state

        self.action_history = []

        self.evaluators = box(evaluators or [])

        self.save_states = save_states if save_states else False
        self.state_history: list[State] = []
        self.move_info = []

    def __repr__(self):
        text = f'{self.agents[0]} vs {self.agents[1]}, first: {self.first}, {self.state.n_pawns} pawns'
        if self.winner is not None:
            text += f', winner: {self.winner} ({self.agents[self.winner]}), {len(self.action_history)} moves'
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
        return self._run()

    def _run(self):
        if not self.action_history:
            last_action = None
            if self.save_states:
                self.state_history.append(self.state.copy())
            while not self.state.game_over():
                for evaluator in self.evaluators:
                    policy, state_value = evaluator.evaluate(self.state)
                    logger.info(
                        f"Evaluation from {evaluator.__class__.__name__}:\n"
                        f"Value: {state_value: .4f}\n"
                        f"Policy: {array2string(policy)}\n"
                    )
                player = self.state.get_cur_player()
                try:
                    action, exe_time = get_timed_action(
                        player=self.agents[player],
                        state=self.state.copy(),
                        last_action=last_action,
                        time_left=self.times_left[player],
                    )
                    logger.info(f'Player {player} action: {action} (in {exe_time:.3f}s)')
                except TimeoutError:
                    self.state.set_timed_out(player)
                    break

                if self.times_left[player]:
                    self.times_left[player] -= exe_time

                self.action_history.append(action)
                self.state.apply_action(action)
                last_action = action
                if self.save_states:
                    self.state_history.append(self.state.copy())
                    self.move_info.append(self.agents[player].get_move_info())
                self._post_apply_action()

            logger.info(f'Game over: {self}')
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

    def to_file(self, filename: str | Path):
        """
        Save the game results on disk

        Note: maybe serialize and save whole objects
        """
        if self.winner is None:
            raise ValueError('Cannot save results to file. Run the game first with run().')
        results = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    @classmethod
    def from_file(cls, filename: str | Path):
        with open(filename, 'r') as f:
            results = json.load(f)
        game = GameFromState(
            state=State(**results['state']),
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
        state = State(n_pawns=n_pawns, first=first)
        super().__init__(state=state, **kwargs)


def handle_timeout(signum, frame):
    """
    Define behavior in case of timeout.
    """
    raise TimeoutError


def get_timed_action(player, state, last_action, time_left):
    """
    Get an action from a player with a timeout.
    """
    if not time_left:
        start_time = time()
        return player.get_action(state, last_action, time_left), time() - start_time

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, time_left)
    start_time = time()
    try:
        action = player.get_action(state, last_action, time_left)
    finally:
        exe_time = time() - start_time
        signal.setitimer(signal.ITIMER_REAL, 0)
    return action, exe_time
