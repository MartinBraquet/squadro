import pygame

from squadro.agent import Agent
from squadro.board import get_position_from_click


class MyAgent(Agent):

	def get_action(self, state, last_action, time_left):
		played = False
		action = None
		while not played:
			for event in pygame.event.get():
				if event.type == pygame.MOUSEBUTTONUP:
					pos = get_position_from_click()
					valid_click = False

					if state.get_cur_player() == 0:
						pawn = pos[0] - 1
						if 0 <= pawn <= state.n_pawns - 1 and state.get_pawn_position(0, pawn)[1] <= pos[1] <= state.get_pawn_position(0, pawn)[1] + 1:
							valid_click = True
					else:
						pawn = pos[1] - 1
						if 0 <= pawn <= state.n_pawns - 1 and state.get_pawn_position(1, pawn)[0] <= pos[0] <= state.get_pawn_position(1, pawn)[0] + 1:
							valid_click = True
					
					if valid_click and state.is_action_valid(pawn):
						played = True
						action = pawn

				if event.type == pygame.QUIT:
					pygame.quit() 
					exit(0)

		return action
  
	def get_name(self):
		return "human_agent"