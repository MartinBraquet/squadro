import argparse
from squadro_state import SquadroState

"""
Runs the game
"""
def main(agent_0, agent_1, first):
	# Initialisation
	cur_state = SquadroState()
	if first != -1:
		cur_state.cur_player = first
	agents = [getattr(__import__(agent_0), 'MyAgent')(), getattr(__import__(agent_1), 'MyAgent')()]
	agents[0].set_id(0)
	agents[1].set_id(1)
	last_action = None

	while not cur_state.game_over():

		# Make move
		cur_player = cur_state.get_cur_player()
		action = get_action_timed(agents[cur_player], cur_state.copy(), last_action)

		if cur_state.is_action_valid(action):
			cur_state.apply_action(action)
			last_action = action
		else:
			cur_state.set_invalid_action(cur_player)
	print(cur_player)
"""
Get an action from player with a timeout.
"""
def get_action_timed(player, state, last_action):
	action = player.get_action(state, last_action, 50)
	return action


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ai0", help="path to the ai that will play as player 0")
	parser.add_argument("-ai1", help="path to the ai that will play as player 1")
	parser.add_argument("-f", help="indicates the player (0 or 1) that plays first; random otherwise")
	args = parser.parse_args()

	ai0 = args.ai0 if args.ai0 != None else "human_agent"
	ai1 = args.ai1 if args.ai1 != None else "human_agent"
	first = int(args.f) if args.f != None else -1

	main(ai0, ai1, first)
