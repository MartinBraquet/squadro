import argparse

import numpy as np
import torch

from squadro.state import State

"""
Runs the game
"""
def main(agent_0, agent_1):

    victory = 0
    for i in range(1000):
        (results, winner) = game(agent_0, agent_1, i)
        victory += 1 - winner # number of victories for player 0 (main)
        print('Victory average for the main model VS other model', i, '(', 1 - winner, ')', ':', "{0:.2f}".format(100 * victory / (i+1)), '%')

def game(agent_0, agent_1, i):
    model_path       = 'model/{}.pt'.format(agent_0)
    other_model_path = 'model/{}.pt'.format(agent_1)
    
    results = 0
    init = 1

    # Initialisation
    cur_state = State()
    
    agents = [getattr(__import__(agent_0), 'MyAgent')(), getattr(__import__(agent_0), 'MyAgent')()]
    
    DNN_main_player = 0 if i % 2 == 0 else 1
    cur_state.cur_player = 0 if (2 * i) % 2 == 0 else 1
    
    print(DNN_main_player, cur_state.cur_player)
    
    agents[0].set_id(0)
    agents[1].set_id(1)
    agents[1 - DNN_main_player].set_model_path(other_model_path)
    '''
    agents[0].epsilonMCTS = 0
    agents[1].epsilonMCTS = 0
    '''
    agents[0].epsilon_move = 0
    agents[1].epsilon_move = 0
    if i == 0:
        print('Network 0 (main) -------------------------------------------------------')
        print_network(agents[DNN_main_player].deepnetwork)
        print('Network 1 (other) -------------------------------------------------------')
        print_network(agents[1 - DNN_main_player].deepnetwork)

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
        
        if init:
            results = agents[cur_player].results
            init = 0
        else:
            results = np.append(results, agents[cur_player].results, 0)

    return (results, abs(DNN_main_player - cur_player))

"""
Get an action from player with a timeout.
"""
def get_action_timed(player, state, last_action):
	action = player.get_action(state, last_action, 50)
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
    parser.add_argument("-ai1", help="path to the ai that will play as player 0")
    args = parser.parse_args()
    
    ai0 = args.ai0 if args.ai0 != None else "error"
    ai1 = args.ai1 if args.ai1 != None else "error"

    main(ai0, ai1)
