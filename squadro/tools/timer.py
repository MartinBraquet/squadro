from threading import Thread
from time import time

import pygame


# Timer makes pygame raises this calling pygame.display.flip():
# pygame.error: Could not make GL context current: BadAccess (attempt to access private resource denied)


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


def pretty_print_time(seconds):
    # Calculate minutes and seconds
    minutes, secs = divmod(seconds, 60)
    # Format the time as mm:ss
    return f"{int(minutes):02d}:{int(secs):02d}"
