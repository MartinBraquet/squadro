import argparse
from time import sleep

import pygame

from squadro.board import Board, handle_events, check_quit
from squadro.game import Game
from squadro.squadro_state import SquadroState
from squadro.tools.constants import DefaultParams


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
        board = Board(self.game.n_pawns)
        state = SquadroState(n_pawns=self.game.n_pawns, first=self.game.first)
        board.turn_draw(state)

        action_history = self.game.run()
        previous_states = []

        while True:
            command = self.get_command()
            if command == 'next':
                if state.game_over():
                    continue
                action = action_history[len(previous_states)]
                previous_states.append(state.copy())
                # print(action)
                state.apply_action(action)

            elif command == 'previous':
                if previous_states:
                    state = previous_states.pop()

            board.turn_draw(state)

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


class RealTimeAnimatedGame(Game):
    """
    Runs the game and renders the board in real time
    """

    def __init__(
        self,
        agent_0=None,
        agent_1=None,
        time_out=None,
        **kwargs
    ):
        super().__init__(
            agent_0=agent_0 or 'human',
            agent_1=agent_1 or 'human',
            time_out=time_out or DefaultParams.time_out,
            **kwargs
        )
        self.board = Board(self.n_pawns)
        self.draw()

    def run(self):
        super().run()
        while True:
            self.board.display_winner(self.state)
            handle_events()
            sleep(.5)

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

    RealTimeAnimatedGame(_args.ai0, _args.ai1, _args.t, first=_args.f).run()
