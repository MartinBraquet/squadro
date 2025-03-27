import argparse
import signal
from threading import Thread
from time import sleep, time

import pygame

from squadro.board import Board, handle_events
from squadro.squadro_state import SquadroState
from squadro.tools.constants import DefaultParams
from squadro.tools.utils import get_agent

# Timer makes pygame raises this calling pygame.display.flip():
# pygame.error: Could not make GL context current: BadAccess (attempt to access private resource denied)
# USE_TIMER = True


sleep_seconds = .2


def play_animated_game(
    agent_0=None,
    agent_1=None,
    time_out=None,
    first=None,
    n_pawns=None,
):
    """
    Runs the game
    """
    agent_0 = agent_0 if agent_0 is not None else 'human'
    agent_1 = agent_1 if agent_1 is not None else 'human'
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
    start_time = time()
    try:
        action = player.get_action(state, last_action, time_left)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        exe_time = time() - start_time
    print('action', action)
    return action, exe_time


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

    play_animated_game(_args.ai0, _args.ai1, _args.t, _args.f)
