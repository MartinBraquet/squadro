import pygame
import argparse
from pygame.locals import *
from time import sleep
from squadro_state import SquadroState

def main(agent_0, agent_1, time_out, sleep_time):
	# Initialisation
	pygame.init()
	n_tiles = 7
	n_pawns = 5
	screen = pygame.display.set_mode((n_tiles * 100, n_tiles * 100))

	# Ressourses
	tile = pygame.image.load("../resources/tile.png")
	corner = pygame.image.load("../resources/corner.png")
	start_l = [pygame.image.load("resources/start_" + str(x) + "_l.png") for x in range(1, 4)]
	start_b = [pygame.image.load("resources/start_" + str(x) + "_b.png") for x in range(1, 4)]
	start_r = [pygame.image.load("resources/start_" + str(x) + "_r.png") for x in range(1, 4)]
	start_t = [pygame.image.load("resources/start_" + str(x) + "_t.png") for x in range(1, 4)]
	yellow_pawn = pygame.image.load("../resources/yellow_pawn.png")
	red_pawn = pygame.image.load("../resources/red_pawn.png")
	yellow_pawn_ret = pygame.image.load("../resources/yellow_pawn_ret.png")
	red_pawn_ret = pygame.image.load("../resources/red_pawn_ret.png")
	yellow_pawn_fin = pygame.image.load("../resources/yellow_pawn_fin.png")
	red_pawn_fin = pygame.image.load("../resources/red_pawn_fin.png")

	cur_state = SquadroState()
	agents = [getattr(__import__(agent_0), 'MyAgent')(), getattr(__import__(agent_1), 'MyAgent')()]

	while not cur_state.game_over():
		# Clear screen
		screen.fill(0)

		# Draw the tiles
		for i in range(1, n_tiles - 1):
			for j in range(1, n_tiles - 1):
				screen.blit(tile, (i * 100, j * 100))
		screen.blit(corner, (0, 0))
		screen.blit(start_l[0], (0, 100))
		screen.blit(start_l[2], (0, 200))
		screen.blit(start_l[1], (0, 300))
		screen.blit(start_l[2], (0, 400))
		screen.blit(start_l[0], (0, 500))
		screen.blit(corner, (0, 600))
		screen.blit(start_b[0], (100, 600))
		screen.blit(start_b[2], (200, 600))
		screen.blit(start_b[1], (300, 600))
		screen.blit(start_b[2], (400, 600))
		screen.blit(start_b[0], (500, 600))
		screen.blit(corner, (600, 600))
		screen.blit(start_r[2], (600, 500))
		screen.blit(start_r[0], (600, 400))
		screen.blit(start_r[1], (600, 300))
		screen.blit(start_r[0], (600, 200))
		screen.blit(start_r[2], (600, 100))
		screen.blit(corner, (600, 0))
		screen.blit(start_t[2], (500, 0))
		screen.blit(start_t[0], (400, 0))
		screen.blit(start_t[1], (300, 0))
		screen.blit(start_t[0], (200, 0))
		screen.blit(start_t[2], (100, 0))

		# Draw the pawns
		for i in range(n_pawns):
			if cur_state.is_pawn_finished(0, i):
				screen.blit(yellow_pawn_fin, cur_state.get_pawn_position(0, i))
			elif cur_state.is_pawn_returning(0, i):
				screen.blit(yellow_pawn_ret, cur_state.get_pawn_position(0, i))
			else:
				screen.blit(yellow_pawn, cur_state.get_pawn_position(0, i))
			
			if cur_state.is_pawn_finished(1, i):
				screen.blit(red_pawn_fin, cur_state.get_pawn_position(1, i))
			elif cur_state.is_pawn_returning(1, i):
				screen.blit(red_pawn_ret, cur_state.get_pawn_position(1, i))
			else:
				screen.blit(red_pawn, cur_state.get_pawn_position(1, i))

		# Draw who's turn it is
		font1 = pygame.font.Font("freesansbold.ttf", 12)
		text1 = font1.render("Current player:", True, (255, 255, 255), (34, 34, 34))
		textRect1 = text1.get_rect()
		textRect1.center = ((n_tiles - 1) * 100 + 50, (n_tiles - 1) * 100 + 38)
		screen.blit(text1, textRect1)

		font2 = pygame.font.Font("freesansbold.ttf", 20)
		if cur_state.get_cur_player() == 0:
			text2 = font2.render("    ", True, (255, 164, 0), (255, 164, 0))
		else:
			text2 = font2.render("    ", True, (126, 0, 0), (126, 0, 0))
		textRect2 = text2.get_rect()
		textRect2.center = ((n_tiles - 1) * 100 + 50, (n_tiles - 1) * 100 + 62)
		screen.blit(text2, textRect2)


		# Update screen
		pygame.display.flip()

		# Make move
		action = agents[cur_state.get_cur_player()].get_action(cur_state, False, False)
		cur_state.apply_action(action)


		# Events
		for event in pygame.event.get():

			# Quit when pressing the X button
			if event.type == pygame.QUIT:
				pygame.quit() 
				exit(0) 

		sleep(sleep_time)

	while True:
		# Clear screen
		screen.fill(0)

		# Draw the tiles
		for i in range(1, n_tiles - 1):
			for j in range(1, n_tiles - 1):
				screen.blit(tile, (i * 100, j * 100))
		screen.blit(corner, (0, 0))
		screen.blit(start_l[0], (0, 100))
		screen.blit(start_l[2], (0, 200))
		screen.blit(start_l[1], (0, 300))
		screen.blit(start_l[2], (0, 400))
		screen.blit(start_l[0], (0, 500))
		screen.blit(corner, (0, 600))
		screen.blit(start_b[0], (100, 600))
		screen.blit(start_b[2], (200, 600))
		screen.blit(start_b[1], (300, 600))
		screen.blit(start_b[2], (400, 600))
		screen.blit(start_b[0], (500, 600))
		screen.blit(corner, (600, 600))
		screen.blit(start_r[2], (600, 500))
		screen.blit(start_r[0], (600, 400))
		screen.blit(start_r[1], (600, 300))
		screen.blit(start_r[0], (600, 200))
		screen.blit(start_r[2], (600, 100))
		screen.blit(corner, (600, 0))
		screen.blit(start_t[2], (500, 0))
		screen.blit(start_t[0], (400, 0))
		screen.blit(start_t[1], (300, 0))
		screen.blit(start_t[0], (200, 0))
		screen.blit(start_t[2], (100, 0))

		# Draw the pawns
		for i in range(n_pawns):
			if cur_state.is_pawn_finished(0, i):
				screen.blit(yellow_pawn_fin, cur_state.get_pawn_position(0, i))
			elif cur_state.is_pawn_returning(0, i):
				screen.blit(yellow_pawn_ret, cur_state.get_pawn_position(0, i))
			else:
				screen.blit(yellow_pawn, cur_state.get_pawn_position(0, i))
			
			if cur_state.is_pawn_finished(1, i):
				screen.blit(red_pawn_fin, cur_state.get_pawn_position(1, i))
			elif cur_state.is_pawn_returning(1, i):
				screen.blit(red_pawn_ret, cur_state.get_pawn_position(1, i))
			else:
				screen.blit(red_pawn, cur_state.get_pawn_position(1, i))

		# Print the winner
		font = pygame.font.Font("freesansbold.ttf", 48)

		if cur_state.get_winner() == 0:
			text = font.render(" Yellow wins! ", True, (255, 164, 0), (34, 34, 34))
		else:
			text = font.render(" Red wins! ", True, (126, 0, 0), (34, 34, 34))

		textRect = text.get_rect()
		textRect.center = (n_tiles * 100 // 2, n_tiles * 100 // 2)
		screen.blit(text, textRect)

		# Update screen
		pygame.display.flip()

		for event in pygame.event.get():

			if event.type == pygame.QUIT:
				pygame.quit() 
				exit(0) 

		sleep(1)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ai0", help="path to the ai that will play as player 0")
	parser.add_argument("-ai1", help="path to the ai that will play as player 1")
	parser.add_argument("-t", help="total number of seconds credited to each player")
	parser.add_argument("-s", help="modify speed of the game (number of seconds between each  move")
	args = parser.parse_args()

	ai0 = args.ai0 if args.ai0 != None else "human_agent"
	ai1 = args.ai1 if args.ai1 != None else "human_agent"
	time_out = int(args.t) if args.t != None else 5
	sleep_time = int(args.s) if args.s != None else 0

	main(ai0, ai1, time_out, sleep_time)
