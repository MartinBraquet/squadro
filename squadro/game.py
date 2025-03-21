import argparse
import signal
from threading import Thread
from time import sleep, time

import pygame

from squadro_state import SquadroState

"""
Runs the game
"""

# Timer makes pygame raises this calling pygame.display.flip():
# pygame.error: Could not make GL context current: BadAccess (attempt to access private resource denied)
USE_TIMER = False


def main(agent_0='human_agent', agent_1='human_agent', time_out=900, first=-1):
    # Initialisation
    pygame.init()
    n_tiles = 7
    n_pawns = 5
    board = Board(n_tiles, n_pawns)
    cur_state = SquadroState()
    if first != -1:
        cur_state.cur_player = first
    agents = [getattr(__import__(agent_0), 'MyAgent')(), getattr(__import__(agent_1), 'MyAgent')()]
    agents[0].set_id(0)
    agents[1].set_id(1)
    times_left = [time_out, time_out]
    last_action = None

    while not cur_state.game_over():
        # Draw board
        board.screen.fill(0)
        board.draw_board(cur_state)
        board.show_turn(cur_state)

        # Update screen
        pygame.display.flip()

        # Make move
        cur_player = cur_state.get_cur_player()
        timer_stop = [False]
        if USE_TIMER:
            timer = TimerDisplay(board, cur_player, times_left.copy(), timer_stop)
            timer.start()
        try:
            action, exe_time = get_action_timed(agents[cur_player], cur_state.copy(), last_action,
                                                times_left[cur_player])
            timer_stop[0] = True
            times_left[cur_player] -= exe_time

            if cur_state.is_action_valid(action):
                cur_state.apply_action(action)
                last_action = action
            else:
                cur_state.set_invalid_action(cur_player)

        except TimeoutError:
            timer_stop[0] = True
            cur_state.set_timed_out(cur_player)

        if USE_TIMER:
            timer.join()

        # Events
        for event in pygame.event.get():

            # Quit when pressing the X button
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

    pygame.quit()
    exit(0)

    # Game finished: display the winner
    while True:
        # Draw board
        board.screen.fill(0)
        board.draw_board(cur_state)

        # Print the winner
        font = pygame.font.Font("freesansbold.ttf", 48)

        if cur_state.get_winner() == 0:
            text = font.render(" Yellow wins! ", True, (255, 164, 0), (34, 34, 34))
        else:
            text = font.render(" Red wins! ", True, (150, 0, 0), (34, 34, 34))

        textRect = text.get_rect()
        textRect.center = (n_tiles * 100 // 2, n_tiles * 100 // 2)
        board.screen.blit(text, textRect)

        # Print if time-out or invalid action
        if cur_state.timeout_player != None:
            font2 = pygame.font.Font("freesansbold.ttf", 18)
            text2 = font2.render("The opponent timed out", True, (255, 255, 255), (34, 34, 34))
            textRect2 = text2.get_rect()
            textRect2.center = (n_tiles * 100 // 2, n_tiles * 100 // 2 + 34)
            board.screen.blit(text2, textRect2)
        elif cur_state.invalid_player != None:
            font2 = pygame.font.Font("freesansbold.ttf", 18)
            text2 = font2.render("The opponent made an invalid move", True, (255, 255, 255), (34, 34, 34))
            textRect2 = text2.get_rect()
            textRect2.center = (n_tiles * 100 // 2, n_tiles * 100 // 2 + 34)
            board.screen.blit(text2, textRect2)

        # Update screen
        pygame.display.flip()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

        sleep(1)


"""
Define behavior in case of timeout.
"""


def handle_timeout(signum, frame):
    raise TimeoutError()


"""
Get an action from player with a timeout.
"""


def get_action_timed(player, state, last_action, time_left):
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, time_left)
    exe_time = time()
    try:
        action = player.get_action(state, last_action, time_left)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        exe_time = time() - exe_time
    return action, exe_time


class Board:

    def __init__(self, n_tiles, n_pawns):
        # Initialise screen
        self.n_tiles = n_tiles
        self.n_pawns = n_pawns
        self.screen = pygame.display.set_mode((n_tiles * 100, n_tiles * 100))

        # Ressourses
        self.tile = pygame.image.load("../resources/tile.png")
        self.corner = pygame.image.load("../resources/corner.png")
        self.start_l = [pygame.image.load("resources/start_" + str(x) + "_l.png") for x in range(1, 4)]
        self.start_b = [pygame.image.load("resources/start_" + str(x) + "_b.png") for x in range(1, 4)]
        self.start_r = [pygame.image.load("resources/start_" + str(x) + "_r.png") for x in range(1, 4)]
        self.start_t = [pygame.image.load("resources/start_" + str(x) + "_t.png") for x in range(1, 4)]
        self.yellow_pawn = pygame.image.load("../resources/yellow_pawn.png")
        self.red_pawn = pygame.image.load("../resources/red_pawn.png")
        self.yellow_pawn_ret = pygame.image.load(
            "../resources/yellow_pawn_ret.png")
        self.red_pawn_ret = pygame.image.load("../resources/red_pawn_ret.png")
        self.yellow_pawn_fin = pygame.image.load(
            "../resources/yellow_pawn_fin.png")
        self.red_pawn_fin = pygame.image.load("../resources/red_pawn_fin.png")

    def draw_board(self, cur_state):
        # Draw the tiles
        for i in range(1, self.n_tiles - 1):
            for j in range(1, self.n_tiles - 1):
                self.screen.blit(self.tile, (i * 100, j * 100))
        self.screen.blit(self.corner, (0, 0))
        self.screen.blit(self.start_l[0], (0, 100))
        self.screen.blit(self.start_l[2], (0, 200))
        self.screen.blit(self.start_l[1], (0, 300))
        self.screen.blit(self.start_l[2], (0, 400))
        self.screen.blit(self.start_l[0], (0, 500))
        self.screen.blit(self.corner, (0, 600))
        self.screen.blit(self.start_b[0], (100, 600))
        self.screen.blit(self.start_b[2], (200, 600))
        self.screen.blit(self.start_b[1], (300, 600))
        self.screen.blit(self.start_b[2], (400, 600))
        self.screen.blit(self.start_b[0], (500, 600))
        self.screen.blit(self.corner, (600, 600))
        self.screen.blit(self.start_r[2], (600, 500))
        self.screen.blit(self.start_r[0], (600, 400))
        self.screen.blit(self.start_r[1], (600, 300))
        self.screen.blit(self.start_r[0], (600, 200))
        self.screen.blit(self.start_r[2], (600, 100))
        self.screen.blit(self.corner, (600, 0))
        self.screen.blit(self.start_t[2], (500, 0))
        self.screen.blit(self.start_t[0], (400, 0))
        self.screen.blit(self.start_t[1], (300, 0))
        self.screen.blit(self.start_t[0], (200, 0))
        self.screen.blit(self.start_t[2], (100, 0))

        # Draw the pawns
        for i in range(self.n_pawns):
            if cur_state.is_pawn_finished(0, i):
                self.screen.blit(self.yellow_pawn_fin, cur_state.get_pawn_position(0, i))
            elif cur_state.is_pawn_returning(0, i):
                self.screen.blit(self.yellow_pawn_ret, cur_state.get_pawn_position(0, i))
            else:
                self.screen.blit(self.yellow_pawn, cur_state.get_pawn_position(0, i))

            if cur_state.is_pawn_finished(1, i):
                self.screen.blit(self.red_pawn_fin, cur_state.get_pawn_position(1, i))
            elif cur_state.is_pawn_returning(1, i):
                self.screen.blit(self.red_pawn_ret, cur_state.get_pawn_position(1, i))
            else:
                self.screen.blit(self.red_pawn, cur_state.get_pawn_position(1, i))

    def show_turn(self, cur_state):
        # Draw who's turn it is
        font1 = pygame.font.Font("freesansbold.ttf", 12)
        text1 = font1.render("Current player:", True, (255, 255, 255), (34, 34, 34))
        textRect1 = text1.get_rect()
        textRect1.center = ((self.n_tiles - 1) * 100 + 50, (self.n_tiles - 1) * 100 + 38)
        self.screen.blit(text1, textRect1)

        font2 = pygame.font.Font("freesansbold.ttf", 20)
        if cur_state.get_cur_player() == 0:
            text2 = font2.render("    ", True, (255, 164, 0), (255, 164, 0))
        else:
            text2 = font2.render("    ", True, (150, 0, 0), (150, 0, 0))
        textRect2 = text2.get_rect()
        textRect2.center = ((self.n_tiles - 1) * 100 + 50, (self.n_tiles - 1) * 100 + 62)
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
        text = font.render("  " + str(int(times_left[0])) + " sec  ", True, (255, 255, 255), (34, 34, 34))
        textRect = text.get_rect()
        textRect.center = (50, (self.n_tiles - 1) * 100 + 62)
        self.screen.blit(text, textRect)
        text = font.render("  " + str(int(times_left[1])) + " sec  ", True, (255, 255, 255), (34, 34, 34))
        textRect = text.get_rect()
        textRect.center = ((self.n_tiles - 1) * 100 + 50, 62)
        self.screen.blit(text, textRect)


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


def run():
    main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ai0", help="path to the ai that will play as player 0")
    parser.add_argument("-ai1", help="path to the ai that will play as player 1")
    parser.add_argument("-t", help="time out: total number of seconds credited to each AI player")
    parser.add_argument("-f", help="indicates the player (0 or 1) that plays first; random otherwise")
    args = parser.parse_args()

    ai0 = args.ai0 if args.ai0 is not None else "human_agent"
    ai1 = args.ai1 if args.ai1 is not None else "human_agent"
    time_out = float(args.t) if args.t is not None else 900.0
    first = int(args.f) if args.f is not None else -1

    main(ai0, ai1, time_out, first)
