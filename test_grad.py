import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(3, 2)
        
    def forward(self, x):
        #print(x.shape)
        l = len(x.size())
        
        if l == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.lin(x))

        return x
    
    
if __name__ == "__main__":

    network = DeepNetwork()
   
    optimizer = optim.SGD(network.parameters(), lr=1, weight_decay=0.0001, momentum=0.9) # regularization done with weight_decay
    lMSE = nn.MSELoss()

    network.train()
    
    x = torch.from_numpy(np.array([1,1,1])).float()
    z = torch.from_numpy(np.array([100,100])).float()

    y = network(x)  # Infer a batch through the network
    
    print(x)
    print(y)
    print(z)
    
    loss = lMSE(z, y)
    
    print(loss)
    
    print(network.lin.weight)
    
    optimizer.zero_grad()  # (Re)Set all the gradients to zero
    loss.backward()  # Compute the backward pass based on the gradients and activations
    
    print(network.lin.weight.grad)
    print(network.lin.weight)
    
    optimizer.step()  # Update the weights
    print(network.lin.weight)
            
    network.eval()
    
    y = network(x)  # Infer a batch through the network
    
    print(network.lin.weight.grad)
      
    print(y)