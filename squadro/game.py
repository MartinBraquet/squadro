import argparse
import signal
from pathlib import Path
from threading import Thread
from time import sleep, time

import pygame

from squadro.squadro_state import SquadroState, get_moves
from squadro.tools.constants import DefaultParams
from squadro.tools.utils import get_agent

RESOURCE_PATH = Path(__file__).parent / 'resources'

# Timer makes pygame raises this calling pygame.display.flip():
# pygame.error: Could not make GL context current: BadAccess (attempt to access private resource denied)
# USE_TIMER = True


sleep_seconds = .2


def run(
    agent_0=None,
    agent_1=None,
    time_out=None,
    first=None,
    n_pawns=None,
):
    """
    Runs the game
    """
    agent_0 = agent_0 if agent_0 is not None else DefaultParams.agent
    agent_1 = agent_1 if agent_1 is not None else DefaultParams.agent
    time_out = float(time_out) if time_out is not None else DefaultParams.time_out
    # first = int(first) if first is not None else DefaultParams.first
    n_pawns = int(n_pawns) if n_pawns is not None else DefaultParams.n_pawns

    # Initialisation
    pygame.init()

    board = Board(n_pawns)
    state = SquadroState(n_pawns=n_pawns, first=first)
    agents = [get_agent(a) for a in (agent_0, agent_1)]
    agents[0].set_id(0)
    agents[1].set_id(1)
    times_left = [time_out] * 2
    last_action = None

    while not state.game_over():
        # Draw board
        board.screen.fill(0)
        board.draw_board(state)
        board.show_turn(state)
        board.show_timer(times_left)

        # Update screen
        pygame.display.flip()

        # Make move
        cur_player = state.get_cur_player()
        # timer_stop = [False]
        # if USE_TIMER:
        #     timer = TimerDisplay(board, cur_player, times_left.copy(),
        #                          timer_stop)
        #     timer.start()
        try:
            action, exe_time = get_action_timed(
                agents[cur_player],
                state.copy(),
                last_action,
                times_left[cur_player]
            )
            # timer_stop[0] = True
            times_left[cur_player] -= exe_time

            if state.is_action_valid(action):
                state.apply_action(action)
                last_action = action
            else:
                state.set_invalid_action(cur_player)

        except TimeoutError:
            # timer_stop[0] = True
            state.set_timed_out(cur_player)

        # if USE_TIMER:
        #     timer.join()

        handle_events()

    # Game finished: display the winner
    while True:
        # Draw board
        board.screen.fill(0)
        board.draw_board(state)
        board.display_winner(state)

        # Update screen
        pygame.display.flip()

        handle_events()

        sleep(sleep_seconds)


def handle_events():
    # Events
    for event in pygame.event.get():
        # Quit when pressing the X button
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)


def handle_timeout(signum, frame):
    """
    Define behavior in case of timeout.
    """
    raise TimeoutError()


def get_action_timed(player, state, last_action, time_left):
    """
    Get an action from player with a timeout.
    """
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, time_left)
    exe_time = time()
    try:
        action = player.get_action(state, last_action, time_left)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        exe_time = time() - exe_time
    print('action', action)
    return action, exe_time


class Board:

    def __init__(self, n_pawns, n_tiles=None):
        # Initialise screen
        self.n_pawns = n_pawns
        self.n_tiles = n_tiles if n_tiles is not None else n_pawns + 2
        self.screen = pygame.display.set_mode(
            (self.n_tiles * 100, self.n_tiles * 100),
            # pygame.RESIZABLE,
            # pygame.FULLSCREEN
        )

        # Resources
        self.tile = pygame.image.load(RESOURCE_PATH / "tile.png")
        self.corner = pygame.image.load(RESOURCE_PATH / "corner.png")
        self.start_l = [
            pygame.image.load(RESOURCE_PATH / ("start_" + str(x) + "_l.png"))
            for x in range(1, 4)
        ]
        self.start_b = [
            pygame.image.load(RESOURCE_PATH / ("start_" + str(x) + "_b.png"))
            for x in range(1, 4)
        ]
        self.start_r = [
            pygame.image.load(RESOURCE_PATH / ("start_" + str(x) + "_r.png"))
            for x in range(1, 4)
        ]
        self.start_t = [
            pygame.image.load(RESOURCE_PATH / ("start_" + str(x) + "_t.png"))
            for x in range(1, 4)
        ]
        self.yellow_pawn = pygame.image.load(RESOURCE_PATH / "yellow_pawn.png")
        self.red_pawn = pygame.image.load(RESOURCE_PATH / "red_pawn.png")
        self.yellow_pawn_ret = pygame.image.load(
            RESOURCE_PATH / "yellow_pawn_ret.png"
        )
        self.red_pawn_ret = pygame.image.load(
            RESOURCE_PATH / "red_pawn_ret.png"
        )
        self.yellow_pawn_fin = pygame.image.load(
            RESOURCE_PATH / "yellow_pawn_fin.png"
        )
        self.red_pawn_fin = pygame.image.load(
            RESOURCE_PATH / "red_pawn_fin.png"
        )

    def draw_board(self, cur_state):
        # Draw the tiles
        for i in range(1, self.n_tiles - 1):
            for j in range(1, self.n_tiles - 1):
                self.screen.blit(self.tile, (i * 100, j * 100))

        n_pixels = (self.n_pawns + 1) * 100

        self.screen.blit(self.corner, (0, 0))
        self.screen.blit(self.corner, (0, n_pixels))
        self.screen.blit(self.corner, (n_pixels, n_pixels))
        self.screen.blit(self.corner, (n_pixels, 0))

        moves = get_moves(self.n_pawns)

        for i in range(self.n_pawns):
            self.screen.blit(self.start_l[moves[0][i] - 1], (0, 100 * (i + 1)))
            self.screen.blit(self.start_b[moves[0][i] - 1],
                             (100 * (i + 1), n_pixels))
            self.screen.blit(self.start_r[moves[1][i] - 1],
                             (n_pixels, 100 * (i + 1)))
            self.screen.blit(self.start_t[moves[1][i] - 1], (100 * (i + 1), 0))

        # Draw the pawns
        for i in range(self.n_pawns):
            if cur_state.is_pawn_finished(0, i):
                self.screen.blit(self.yellow_pawn_fin,
                                 cur_state.get_pawn_position(0, i))
            elif cur_state.is_pawn_returning(0, i):
                self.screen.blit(self.yellow_pawn_ret,
                                 cur_state.get_pawn_position(0, i))
            else:
                self.screen.blit(self.yellow_pawn,
                                 cur_state.get_pawn_position(0, i))

            if cur_state.is_pawn_finished(1, i):
                self.screen.blit(self.red_pawn_fin,
                                 cur_state.get_pawn_position(1, i))
            elif cur_state.is_pawn_returning(1, i):
                self.screen.blit(self.red_pawn_ret,
                                 cur_state.get_pawn_position(1, i))
            else:
                self.screen.blit(self.red_pawn,
                                 cur_state.get_pawn_position(1, i))

    def show_turn(self, state):
        # Draw whose turn it is
        font1 = pygame.font.Font("freesansbold.ttf", 12)
        text1 = font1.render("Current player:", True, (255, 255, 255),
                             (34, 34, 34))
        textRect1 = text1.get_rect()
        textRect1.center = (
            (self.n_tiles - 1) * 100 + 50, (self.n_tiles - 1) * 100 + 38)
        self.screen.blit(text1, textRect1)

        font2 = pygame.font.Font("freesansbold.ttf", 20)
        color = (255, 164, 0) if state.get_cur_player() == 0 else (150, 0, 0)
        text2 = font2.render("    ", True, color, color)
        textRect2 = text2.get_rect()
        textRect2.center = (
            (self.n_tiles - 1) * 100 + 50,
            (self.n_tiles - 1) * 100 + 62
        )
        self.screen.blit(text2, textRect2)

        font3 = pygame.font.Font("freesansbold.ttf", 12)
        text3 = font3.render("Time left:", True, (255, 255, 255), (34, 34, 34))
        textRect3 = text3.get_rect()
        textRect3.center = (50, (self.n_tiles - 1) * 100 + 38)
        self.screen.blit(text3, textRect3)

        font3 = pygame.font.Font("freesansbold.ttf", 12)
        text3 = font3.render("Time left:", True, (255, 255, 255), (34, 34, 34))
        textRect3 = text3.get_rect()
        textRect3.center = ((self.n_tiles - 1) * 100 + 50, 38)
        self.screen.blit(text3, textRect3)

        font4 = pygame.font.Font("freesansbold.ttf", 2)
        text4 = font4.render(" " * 60, True, (255, 255, 255), (255, 164, 0))
        textRect4 = text4.get_rect()
        textRect4.center = (50, (self.n_tiles - 1) * 100 + 50)
        self.screen.blit(text4, textRect4)

        text4 = font4.render(" " * 60, True, (255, 255, 255), (150, 0, 0))
        textRect4 = text4.get_rect()
        textRect4.center = ((self.n_tiles - 1) * 100 + 50, 50)
        self.screen.blit(text4, textRect4)

    def show_timer(self, times_left):
        font = pygame.font.Font("freesansbold.ttf", 12)
        text = font.render("  " + str(int(times_left[0])) + " sec  ", True,
                           (255, 255, 255), (34, 34, 34))
        textRect = text.get_rect()
        textRect.center = (50, (self.n_tiles - 1) * 100 + 62)
        self.screen.blit(text, textRect)
        text = font.render("  " + str(int(times_left[1])) + " sec  ", True,
                           (255, 255, 255), (34, 34, 34))
        textRect = text.get_rect()
        textRect.center = ((self.n_tiles - 1) * 100 + 50, 62)
        self.screen.blit(text, textRect)

    def display_winner(self, state):
        """Print the winner"""
        font = pygame.font.Font("freesansbold.ttf", 48)

        if state.get_winner() == 0:
            text = font.render(" Yellow wins! ", True, (255, 164, 0),
                               (34, 34, 34))
        else:
            text = font.render(" Red wins! ", True, (150, 0, 0), (34, 34, 34))

        textRect = text.get_rect()
        textRect.center = (self.n_tiles * 100 // 2, self.n_tiles * 100 // 2)
        self.screen.blit(text, textRect)

        # Print if time-out or invalid action
        if state.timeout_player is not None:
            font2 = pygame.font.Font("freesansbold.ttf", 18)
            text2 = font2.render("The opponent timed out", True,
                                 (255, 255, 255), (34, 34, 34))
            textRect2 = text2.get_rect()
            textRect2.center = (self.n_tiles * 100 // 2, self.n_tiles * 100 // 2 + 34)
            self.screen.blit(text2, textRect2)
        elif state.invalid_player is not None:
            font2 = pygame.font.Font("freesansbold.ttf", 18)
            text2 = font2.render("The opponent made an invalid move", True,
                                 (255, 255, 255), (34, 34, 34))
            textRect2 = text2.get_rect()
            textRect2.center = (self.n_tiles * 100 // 2, self.n_tiles * 100 // 2 + 34)
            self.screen.blit(text2, textRect2)


class TimerDisplay(Thread):

    def __init__(self, board, cur_player, times_left, stopped):
        Thread.__init__(self)
        self.board = board
        self.cur_player = cur_player
        self.times_left = times_left
        self.stopped = stopped

    def run(self):
        beg_time = time()
        delta = self.times_left[self.cur_player] % 1
        # self.board.show_timer(self.times_left)
        pygame.display.flip()

        while not self.stopped[0] and self.times_left[self.cur_player] > 0:
            if time() - beg_time >= delta:
                self.times_left[self.cur_player] -= 1
                # self.board.show_timer(self.times_left)
                pygame.display.flip()
                delta += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ai0",
                        help="path to the ai that will play as player 0")
    parser.add_argument("-ai1",
                        help="path to the ai that will play as player 1")
    parser.add_argument("-t",
                        help="time out: total number of seconds credited to each AI player")
    parser.add_argument("-f",
                        help="indicates the player (0 or 1) that plays first; random otherwise")
    _args = parser.parse_args()

    run(_args.ai0, _args.ai1, _args.t, _args.f)
