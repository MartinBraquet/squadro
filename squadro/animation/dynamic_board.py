from time import sleep

import pygame

from squadro.agents.human_agent import get_action_from_click
from squadro.animation.board import Board, check_quit, get_position_from_click
from squadro.state.evaluators.base import Evaluator
from squadro.state.state import State
from squadro.tools.constants import DefaultParams
from squadro.tools.logs import logger


class DynamicBoardAnimation:
    def __init__(
        self,
        state: State = None,
        n_pawns: int = None,
        evaluators: list[Evaluator] = None,
    ):
        self.state = state or State(n_pawns=n_pawns or DefaultParams.n_pawns)
        self.evaluators = evaluators or []

    @property
    def n_pawns(self) -> int:
        return self.state.n_pawns

    def show(self):
        board = Board(self.state.n_pawns, title=f"Dynamic Board")
        board.turn_draw(self.state)

        previous_states = []

        while True:
            command = self.get_command()
            if isinstance(command, list):
                if len(command) == 1:
                    action = command[0]
                    previous_states.append(self.state)
                    self.state = self.state.get_next_state(action)
                elif len(command) == 2:
                    player, pawn = command
                    previous_states.append(self.state)
                    self.state = self.state.copy()
                    self.state.returning[player][pawn] = not self.state.returning[player][pawn]
                    self.state.finished[player][pawn] = (
                        self.state.returning[player][pawn]
                        and self.state.pos[player][pawn] == self.state.max_pos
                    )
                    self.state.winner = None
                    self.state.game_over_check()
                elif len(command) == 3:
                    player, pawn, pawn_pos = command
                    previous_states.append(self.state)
                    self.state = self.state.copy()
                    self.state.pos[player][pawn] = pawn_pos
                    if pawn_pos == 0:
                        self.state.returning[player][pawn] = True
                    self.state.finished[player][pawn] = (
                        self.state.returning[player][pawn]
                        and self.state.pos[player][pawn] == self.state.max_pos
                    )
                    self.state.winner = None
                    self.state.game_over_check()
                else:
                    raise NotImplementedError

            elif command == 'change_player':
                self.state.cur_player = 1 - self.state.cur_player

            elif command == 'previous':
                if previous_states:
                    self.state = previous_states.pop()

            elif command == 'quit':
                break

            logger.info(f"State: {self.state}")
            for evaluator in self.evaluators:
                policy, state_value = evaluator.evaluate(self.state)
                logger.info(
                    f"Evaluation from {evaluator.__class__.__name__}: {state_value: .4f}\n"
                    f"policy: {policy}\n"
                )

            board.turn_draw(self.state)

            # if self.state.game_over():
            #     board.display_winner(self.state)

    def get_command(self):
        """
        Wait for a command that will dictate the next update in the animation.
        Available commands:
        - next: show the next move
        - previous: show the previous move
        - quit: quit the game
        """
        dragging = None
        try:
            while True:
                for event in pygame.event.get():
                    check_quit(event)

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            sleep(.1)
                            return 'previous'

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            x, y = get_position_from_click()
                            if x - 1 in range(self.n_pawns) and self.state.pos[0][x - 1] == y:
                                dragging = [0, x]
                            if y - 1 in range(self.n_pawns) and self.state.pos[1][y - 1] == x:
                                dragging = [1, y]
                            logger.debug(f'Left click DOWN: {(x, y)}, (player, pawn) = {dragging}')

                        if event.button == 2:
                            action = get_action_from_click(self.state)
                            logger.debug(f'Middle click, {action=}')
                            if action is not None:
                                return [action]

                        if event.button == 3:
                            # Right click
                            x, y = get_position_from_click()
                            logger.debug(f'Right click: {(x, y)}')
                            if x - 1 in range(self.n_pawns) and self.state.pos[0][x - 1] == y:
                                return [0, x - 1]
                            if y - 1 in range(self.n_pawns) and self.state.pos[1][y - 1] == x:
                                return [1, y - 1]
                            if x == y == self.state.max_pos:
                                return 'change_player'

                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1 and dragging:
                            player, start = dragging
                            dragging = None
                            x, y = get_position_from_click()
                            logger.debug(f'Left click UP: {(x, y)}')
                            if player == 0 and start == x and y in range(self.state.max_pos + 1):
                                return [0, x - 1, y]
                            if player == 1 and start == y and x in range(self.state.max_pos + 1):
                                return [1, y - 1, x]

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    return 'previous'

        except SystemExit:
            return 'quit'
