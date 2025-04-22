import pygame

from squadro.agents.agent import Agent
from squadro.animation.board import get_position_from_click, check_quit


def get_action_from_click(state) -> int | None:
    click_pos = get_position_from_click()
    player = state.get_cur_player()
    pawn = click_pos[player] - 1
    if 0 <= pawn <= state.n_pawns - 1:
        pawn_pos = state.get_pawn_position(player, pawn)
        if pawn_pos == click_pos and state.is_action_valid(pawn):
            return pawn
        return None
    return None


class HumanAgent(Agent):

    def get_action(self, state, last_action=None, time_left=None):
        while True:
            for event in pygame.event.get():
                check_quit(event)
                if event.type == pygame.MOUSEBUTTONUP:
                    pawn = get_action_from_click(state)
                    if pawn is not None:
                        return pawn

    @classmethod
    def get_name(cls):
        return "human"
