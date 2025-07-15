from time import time

import pygame

from squadro.tools.constants import PYGAME_REFRESH_SECONDS


def refresh_pygame_window():
    pygame.event.pump()


class PygameRefresher:
    def __init__(self):
        self.pygame_refresh_time = time()
        self.is_pygame_init = pygame.get_init()

    def refresh(self):
        if self.is_pygame_init and time() - self.pygame_refresh_time > PYGAME_REFRESH_SECONDS:
            refresh_pygame_window()
            self.pygame_refresh_time = time()
