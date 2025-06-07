import argparse
from time import sleep

import pygame

from squadro.agents.agent import Agent
from squadro.agents.best import get_best_real_time_game_agent
from squadro.animation.board import Board, handle_events, check_quit
from squadro.game import Game
from squadro.state import State, get_next_state
from squadro.tools.constants import DefaultParams, DATA_PATH
from squadro.tools.dates import get_now
from squadro.tools.disk import mkdir
from squadro.tools.log import logger


class GameAnimation:
    """
    Visualize a game that was played between two computer agents
    """

    def __init__(self, game: Game, move_delay=.05):
        """
        :param game: Game
        :param move_delay: delay in seconds between moves
        """
        self.game = game
        self.move_delay = move_delay

    def show(self):
        board = Board(self.game.n_pawns, title=f"Game Visualization: {self.game.title}")
        state = State(n_pawns=self.game.n_pawns, first=self.game.first)
        board.turn_draw(state)

        action_history = self.game.run()
        previous_states = []

        while True:
            command = self.get_command()
            if command == 'next':
                if state.game_over():
                    continue
                action = action_history[len(previous_states)]
                previous_states.append(state)
                logger.debug(f"{action=}")
                state = get_next_state(state, action)

            elif command == 'previous':
                if previous_states:
                    state = previous_states.pop()

            elif command == 'quit':
                break

            board.turn_draw(state)
            logger.info(f"State: {state.to_list()}")

            if state.game_over():
                board.display_winner(state)

            sleep(self.move_delay)

    @staticmethod
    def get_command():
        """
        Wait for a command that will dictate the next update in the animation.
        Available commands:
        - next: show the next move
        - previous: show the previous move
        - quit: quit the game
        """
        try:
            while True:
                for event in pygame.event.get():
                    check_quit(event)

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RIGHT:
                            sleep(.1)
                            return 'next'
                        if event.key == pygame.K_LEFT:
                            sleep(.1)
                            return 'previous'

                keys = pygame.key.get_pressed()
                if keys[pygame.K_RIGHT]:
                    return 'next'
                if keys[pygame.K_LEFT]:
                    return 'previous'
        except SystemExit:
            return 'quit'


class GamePlay(Game):
    """
    Runs the game and renders the board in real time
    """

    def __init__(
        self,
        agent_0=None,
        agent_1=None,
        time_out=None,
        agent_kwargs: dict = None,
        **kwargs
    ):
        agent_kwargs = agent_kwargs or {}

        agents = dict(agent_0=agent_0, agent_1=agent_1)
        for agent_name, agent in agents.items():
            if agent is None:
                agents[agent_name] = 'human'
            elif agent == 'best':
                agents[agent_name] = get_best_real_time_game_agent(n_pawns=kwargs.get('n_pawns'))
            elif isinstance(agent, Agent):
                for k, v in agent_kwargs.items():
                    setattr(agent, k, v)

        agent_kwargs.setdefault('max_time_per_move', DefaultParams.max_time_per_move_real_time)

        super().__init__(
            **agents,
            time_out=time_out or DefaultParams.time_out,
            agent_kwargs=agent_kwargs,
            **kwargs,
        )
        self.board = Board(self.n_pawns, title=f"Real-Time Game: {self.title}")
        self.draw()

    def run(self):
        try:
            super().run()

            path = DATA_PATH / 'game_results'
            mkdir(path)
            self.to_file(path / f'{get_now()}.json')

            while True:
                self.board.display_winner(self.state)
                self._handle_game_over()
                sleep(.5)
        except SystemExit:
            pass

    @staticmethod
    def _handle_game_over():
        handle_events()

    def _post_apply_action(self):
        self.draw()
        handle_events()

    def draw(self):
        self.board.turn_draw(self.state)
        self.board.show_timer(self.times_left)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ai0",
                        help="path to the AI that will play as player 0")
    parser.add_argument("-ai1",
                        help="path to the AI that will play as player 1")
    parser.add_argument("-t",
                        help="time out: total number of seconds credited to each AI player")
    parser.add_argument("-f",
                        help="indicates the player (0 or 1) that plays first; random otherwise")
    _args = parser.parse_args()

    GamePlay(_args.ai0, _args.ai1, _args.t, first=_args.f).run()
