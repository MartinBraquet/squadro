import argparse

import torch

from squadro.squadro_state import SquadroState

# text = 'model200neurons_7layers'

"""
Runs the game
"""
def main(agent_0, agent_1, path):
    victory = 0
    for i in range(1000):
        _, winner = game(agent_0, agent_1, path, i)
        victory += 1 - winner # number of victories for player 0 (main)
        print('Victory average for the DNN model', path, 'VS other', agent_1, 'model', i, '(', 1 - winner, ')', ':', "{0:.2f}".format(100 * victory / (i+1)), '%')

def game(agent_0, agent_1, path, i):
    
    model_path = 'model/{}.pt'.format(path)
    results = 0
    init = 1

    # Initialisation
    cur_state = SquadroState()
    cur_state.cur_player = 0 if i % 4 < 2 else 1
    
    if i % 2 == 0:
        agents = [getattr(__import__(agent_0), 'MyAgent')(), getattr(__import__(agent_1), 'MyAgent')()]
        DNN_player = 0
    else:
        agents = [getattr(__import__(agent_1), 'MyAgent')(), getattr(__import__(agent_0), 'MyAgent')()]
        DNN_player = 1
    
    print(DNN_player, cur_state.cur_player)
        
    agents[0].set_id(0)
    agents[1].set_id(1)
    agents[DNN_player].epsilonMove = 0
    agents[DNN_player].set_model_path(model_path)
    #if i == 0:
        #print('Network 0 (main) -------------------------------------------------------')
        #print_network(agents[0].deepnetwork)
        #print('Network 1 (other) -------------------------------------------------------')
        #print_network(agents[1].deepnetwork)

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

    print('Last state:', agents[DNN_player].results, 'DNN:', DNN_player, 'win:', cur_player)
    return (0, abs(cur_player - DNN_player))

"""
Get an action from player with a timeout.
"""
def get_action_timed(player, state, last_action):
	action = player.get_action(state, last_action, 5)
	return action

def print_network(network):
    # Print model's state_dict
    mod_dict = network.state_dict()
    print("Model's state_dict:")
    for param_tensor in mod_dict:
        print(param_tensor, "\t", mod_dict[param_tensor].size())
        #print(mod_dict[param_tensor])
        print(torch.sum(mod_dict[param_tensor]))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-ai0", help="path to the ai that will play as player 0")
    parser.add_argument("-ai1", help="path to the ai that will play as player 1")
    parser.add_argument("-p", help="path to the ai that will play as player 1")
    args = parser.parse_args()
    
    ai0 = args.ai0 if args.ai0 != None else "error"
    ai1 = args.ai1 if args.ai1 != None else "error"
    path = args.p if args.p != None else "error"

    main(ai0, ai1, path)
