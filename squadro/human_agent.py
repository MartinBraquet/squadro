import pygame

from squadro.agent import Agent
from squadro.board import get_position_from_click, check_quit


class MyAgent(Agent):

    def get_action(self, state, last_action, time_left):
        while True:
            for event in pygame.event.get():
                check_quit(event)
                if event.type == pygame.MOUSEBUTTONUP:
                    click_pos = get_position_from_click()
                    player = state.get_cur_player()
                    pawn = click_pos[player] - 1
                    if 0 <= pawn <= state.n_pawns - 1:
                        pawn_pos = state.get_pawn_position(player, pawn)
                        if pawn_pos == click_pos:
                            if state.is_action_valid(pawn):
                                return pawn

    def get_name(self):
        return "human"
