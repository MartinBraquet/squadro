import numpy as np
import torch
import random
import argparse

"""
Runs the game
"""
def main(ai, path):
    model_path = 'model/{}.pt'.format(path)
    agent = getattr(__import__(ai), 'MyAgent')()
    agent.set_model_path(model_path)
    
    print_network(agent.deepnetwork)
    

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
    parser.add_argument("-ai", help="path to the ai that will play as player 1")
    parser.add_argument("-p", help="path to the ai that will play as player 1")
    args = parser.parse_args()
    
    path = args.p if args.p != None else "error"
    ai = args.ai if args.ai != None else "error"

    main(ai, path)
