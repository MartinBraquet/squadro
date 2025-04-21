import numpy as np
import torch

from squadro.training_dnn import training

if __name__ == "__main__":

    text = 'contest_agent4'
    new_model_path = 'model/{}_new.pt'.format(text)
    ai0 = 'contest_agent4'
    
    network = getattr(__import__(ai0), 'DeepNetwork')()
    torch.save(network.state_dict(), new_model_path)
    
    all_results = np.repeat(np.array([0, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 0.8, 0.1, 0.05, 0.025, 0.025, 1]).reshape(1,17), 100, 0)
    while True:
        training(all_results, new_model_path, new_model_path, ai0)

