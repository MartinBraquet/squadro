from squadro_state import SquadroState
import numpy as np
import random
import argparse
from training import training, save_new_model, print_network

n_train = 60
n_valid = 60
n_valid_old = 0
n_valid_smart = 0
verif_prob = 0.55
verif_prob_old = -1
verif_prob_smart = -1

"""
Runs the game
"""
def main(agent):
    #all_results = np.load('data/{}.npy'.format(text))
    #np.save('data/{}.npy'.format(text), all_results)
    
    model_path     = 'model/{}.pt'.format(agent)
    new_model_path = 'model/{}_new.pt'.format(agent)
    old_model_path = 'model/{}_old.pt'.format(agent)
    
    while True:
        all_results = np.zeros((1,17)) 
        
        for i in range(n_train):
            print('Training:', i)
            (results, winner) = game(agent, agent, 0, i, model_path, model_path)
            new_result = np.append(results, np.transpose(2 * np.abs(np.transpose(np.reshape(results[:,0], newshape=[-1,1])) + np.transpose(winner * np.ones((np.size(results, 0), 1))) -1) - 1), 1)
            #all_results = new_result #if all_results == None  else np.append(all_results, new_result, 0)
            all_results = np.append(all_results, new_result, 0)
            
        #np.save('data/{}.npy'.format(agent), all_results)
        
        training(all_results, model_path, new_model_path, agent)
        
        print("Validation...")
        
        if validation(agent, agent, model_path, new_model_path, n_valid, verif_prob): # Beats the current model
            if validation(agent, agent, old_model_path, new_model_path, n_valid_old, verif_prob_old):  # Beats the old model (initial)
                if validation('smart_agent', agent, 'smart_agent', new_model_path, n_valid_smart, verif_prob_smart):  # Beats the smart agent
                    # model validated, replaced with new one
                    save_new_model(model_path, new_model_path)

def validation(agent, agent2, model_path, model_path2, n_valid, verif_prob):
    victory = 0
    for i in range(n_valid):
        (results, winner) = game(agent, agent2, 1, i, model_path, model_path2)
        victory += winner # number of victories for new player (model_path2)
        print('Validation against', model_path, 'n°',  i, ':', "{0:.2f}".format(100 * victory / (i+1)), '%')
        if i > n_valid / 2 and victory / (i+1) < verif_prob - 0.1:
            break
    if victory / n_valid > verif_prob:
        return True
        #print(victory / n)
    return False

def game(agent, agent2, validation, i, model_path, new_model_path):
    
    results = 0
    init = 1
    DNN_new_player = 0

    # Initialisation
    cur_state = SquadroState()
    cur_state.cur_player = 0 if i % 2 == 0 else 1
    if i % 4 < 2:
        agents = [getattr(__import__(agent2), 'MyAgent')(), getattr(__import__(agent), 'MyAgent')()]
    else:
        agents = [getattr(__import__(agent), 'MyAgent')(), getattr(__import__(agent2), 'MyAgent')()]
    agents[0].set_id(0)
    agents[1].set_id(1)
    
    if validation: # Use different models during validation phase
        DNN_new_player = 0 if i % 4 < 2 else 1 
        agents[DNN_new_player].set_model_path(new_model_path)
        if model_path != 'smart_agent':
            agents[1 - DNN_new_player].set_model_path(model_path)
        print('Main model:', DNN_new_player, ', First player:', cur_state.cur_player)
        # Remove stochastic actions
        #agents[0].epsilonMCTS = 0
        #agents[1].epsilonMCTS = 0
        agents[0].epsilonMove = 0
        agents[1].epsilonMove = 0
        '''
        print('Current network model...............................................')
        print_network(agents[0].deepnetwork)
        print('New network model....................................................')
        print_network(agents[1].deepnetwork)
        '''
        
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
        
        if cur_player == DNN_new_player:
            if init:
                results = agents[cur_player].results
                init = 0
            else:
                results = np.append(results, agents[cur_player].results, 0)
            
    if validation and i == 0:
        print('New model network:')
        print_network(agents[DNN_new_player].deepnetwork)
    #    print(results)
    #    print(cur_player)
    return (results, 1 - abs(DNN_new_player - cur_player))

"""
Get an action from player with a timeout.
"""
def get_action_timed(player, state, last_action):
	action = player.get_action(state, last_action, 5)
	return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ai", help="path")
    args = parser.parse_args()

    ai = args.ai if args.ai != None else "error"
    main(ai)
