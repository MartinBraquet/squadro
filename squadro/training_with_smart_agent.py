import argparse
import numpy as np
import pandas as pd
import time
from ignite.metrics import Loss as MLoss
from ignite.metrics import Accuracy as MAccuracy
import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def main(path, new):
        
    # Set the training parameters
    epochs = 100
    lr = 0.1
    batch_size = 256
    
    # Torchvision contains a link to download the FashionMNIST dataset. Let's first 
    # store the training and test sets.
    
    train_data = SquadroDataset(path ='data/smart_agent_data.csv')
    
    # We now divide the training data in training set and validation set.
    n_train = len(train_data)
    indices = list(range(n_train))
    split = int(n_train - (n_train * 0.1))  # Keep 10% for validation
    train_set = Subset(train_data, indices[:split])
    val_set = Subset(train_data, indices[split:])
    
    # Object where the data should be moved to:
    #   either the CPU memory (aka RAM + cache)
    #   or GPU memory
    #device = torch.device('cuda') # Note: cuda is the name of the technology inside NVIDIA graphic cards
    #network = DeepNetwork().to(device) # Transfer Network on graphic card.
    
    
    model_path = 'model/{}.pt'.format(path)
    out_model_path = model_path
    
    network = getattr(__import__(path), 'DeepNetwork')()
    if new == '0':
        print(new)
        network.load_state_dict(torch.load(model_path))
    #network.eval()
    #for p in network.parameters():
    #    torch.nn.init.normal_(p, mean=0, std=1)
    #print_network(network)
    
    
    #optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0) # regularization done with weight_decay
    optimizer = optim.Adam(network.parameters(), lr=lr) # regularization done with weight_decay
    lMSE = nn.MSELoss()
    lcross = nn.CrossEntropyLoss()
        
    # Load the data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle = True, drop_last=True)
    #val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    
    '''
    # Metrics in order to check how the training's going
    metrics = MetricsList(
        metrics=[MLoss(l1), MAccuracy()],
        names=["Loss", "Accuracy"]
    )
    '''

    
    # Complete Training Loop
    for epoch in range(epochs):
        print(f"--- Starting epoch {epoch}")
        i = 0
        
        # Train the model
        print("Training...")
        #metrics.reset()
        #print(metrics)
        network.train() # Set the network in training mode => weights become trainable aka modifiable
        for batch in train_loader:
            i += 1
            #print(i)
            x, z, pi = batch
            #x, t = x.to(device), t.to(device)
            
            if len(x.data.size()) == 1:
                continue

            optimizer.zero_grad()  # (Re)Set all the gradients to zero

            p, v = network(x)  # Infer a batch through the network
            
            xd, zd, pid, pd, vd = (x.data.numpy(), z.data.numpy(), pi.data.numpy(), p.data.numpy(), v.data.numpy())
            
            '''
            print('=============================')
            print('x :')
            print(x.data.numpy())
            print('z :')
            print(z)
            print('v :')
            print(v)
            print('p :')
            print(p)
            print('pi :')
            print(pi)
            print('=============================')
            '''

            l1 = lMSE(z, v)
            l2 = categorical_cross_entropy(p, pi)
            loss = l1 + l2
            
            #print_network(network)
            
            loss.backward()  # Compute the backward pass based on the gradients and activations
            optimizer.step()  # Update the weights
            
            #print_network(network)
            
            print('Loss: ', (lMSE(z, v).item(), categorical_cross_entropy(p, pi).item()))
            
            #print_network(network, optimizer)

            #metrics.update(y, t)
            
        #metrics.compute("Train")
        '''
        # Validate the model
        print("Validating...")
        with torch.no_grad():
            #metrics.reset()

            network.eval() # Freeze the network >< training mode
            for batch in val_loader:
                x, t = batch
                #x, t = x.to(device), t.to(device)
    
                y, _ = network(x)
                metrics.update(y, t)
            metrics.compute("Validation")
        print(metrics)
        metrics.clear()
        '''
    network.eval()
    torch.save(network.state_dict(), out_model_path)
    #print_network(network)

def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    y_true = torch.clamp(y_true, 1e-9, 1 - 1e-9)
    return (y_true * torch.log(y_true / y_pred)).sum(dim=1).mean()


class MetricsList():
    def __init__(self, metrics, names):
        self.metrics = metrics
        self.names = names
        self.df = pd.DataFrame(columns=names)

    def update(self, logits, labels):
        """
            Updates all the metrics
            - logits: output of the network
            - labels: ground truth
        """
        for metric in self.metrics:
            metric.update((logits, labels))

    def reset(self):
        for metric in self.metrics:
           metric.reset()

    def clear(self):
       self.df = self.df.iloc[0:0]  # Clear Dataframe
    
    def compute(self, mode):
        data = []
        for metric in self.metrics:
           data.append( metric.compute() )
        self.df.loc[mode] = data
    
    def __str__(self):
       return str(self.df)


 
 
class SquadroDataset(Dataset):
    def __init__(self, path):
        # # All the data preperation tasks can be defined here
        # - Deciding the dataset split (train/test/ validate)
        # - Data Transformation methods 
        # - Reading annotation files (CSV/XML etc.)
        # - Prepare the data to read by an index
        
        data = pd.read_csv(path).to_numpy()
        data = data[:3000,:]
        x = data[:,:-1].astype(float)
        t = data[:,-1].astype(int)
        pi = np.zeros((len(t),5))
        for i in range(len(t)):
            pi[i, t[i]] = 1
        xsum = (np.sum(x[:,0:5], 1) - np.min(x[:,0:5], 1)) - (np.sum(x[:,5:10], 1) - np.min(x[:,5:10], 1))
        z = xsum / 24
        z = z.reshape(len(z),1)
        x = np.concatenate((np.zeros((x.shape[0],1)),x),axis=1)
        self.x = np.transpose(torch.from_numpy(x)).float()
        self.z = np.transpose(torch.from_numpy(z)).float()
        self.pi = np.transpose(torch.from_numpy(pi)).float()
        
         
    def __getitem__(self, index):
        # # Returns data and labels
        # - Apply initiated transformations for data
        # - Push data for GPU memory
        # - better to return the data points as dictionary/ tensor 
        return (self.x[:,index], self.z[:,index], self.pi[:,index])
 
    def __len__(self):
        return len(self.z[0,:])
    
    

def print_network(network):
    # Print model's state_dict
    mod_dict = network.state_dict()
    print("Model's state_dict:")
    for param_tensor in mod_dict:
        print(param_tensor, "\t", mod_dict[param_tensor].size())
        print(mod_dict[param_tensor])
        print(torch.sum(mod_dict[param_tensor]))
        
        

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("-p", help="path")
      parser.add_argument("-n", help="new network")
      args = parser.parse_args()

      path = args.p if args.p != None else "error"
      new = args.n if args.n != None else "error"

      main(path, new)
