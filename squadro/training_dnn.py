from shutil import copyfile

import numpy as np
# from contest_agent import DeepNetwork
# from dummyDNN import DeepNetwork
import pandas as pd
# from ignite.metrics import Loss as MLoss
# from ignite.metrics import Accuracy as MAccuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset


def training(data, model_path, new_model_path, agent):
    print("Training...")
    
    train_data = SquadroDataset(data)
    
    # We now divide the training data in training set and validation set.
    n_train = len(train_data)
    #while n_train % batch_size == 1:
    #    batch_size += 1
    #indices = list(range(n_train))
    #split = int(n_train - (n_train * 0.05))  # Keep 10% for validation
    #train_set = Subset(train_data, indices[:split])
    train_set = train_data
    #val_set = Subset(train_data, indices[split:])
    
    # Set the training parameters
    lr = 0.01
    batch_size = 4 # Check with smaller batch
    epochs = 2 #max(5, int(n_train / 10000))
    
    print('Data:', n_train)
    
    # Object where the data should be moved to:
    #   either the CPU memory (aka RAM + cache)
    #   or GPU memory
    #device = torch.device('cuda') # Note: cuda is the name of the technology inside NVIDIA graphic cards
    #network = DeepNetwork().to(device) # Transfer Network on graphic card.
    
    
    network = getattr(__import__(agent), 'DeepNetwork')()
    network.load_state_dict(torch.load(model_path))
    #torch.save(network.state_dict(), new_model_path)
   
    optimizer = optim.SGD(network.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9) # regularization done with weight_decay
    lMSE = nn.MSELoss()
    
    #print('Network before training...........................................')
    #print_network(network)
    
    # Load the data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    #val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    
    '''
    # Metrics in order to check how the training's going
    metrics = MetricsList(
        metrics=[MLoss(criterion), MAccuracy()],
        names=["Loss", "Accuracy"]
    )
    '''
    
    # Complete Training Loop
    for epoch in range(epochs):
        #print(f"--- Starting epoch {epoch}")
        #start = time.time()
        i = 0
        
        # Train the model
        #print("Training...")
        #metrics.reset()
        #print(metrics)
        network.train() # Set the network in training mode => weights become trainable aka modifiable
        for batch in train_loader:
            #i += 1
            #print(i)
            #print(batch)
            x, pi, z = batch
            #x, t = x.to(device), t.to(device)
            
            #if len(x.data.size()) == 1:
            #    continue

            p, v = network(x)  # Infer a batch through the network
            
            #xd, zd, pid, pd, vd = (x.data.numpy(), z.data.numpy(), pi.data.numpy(), p.data.numpy(), v.data.numpy())
            
            #zprint(pd)
            #print(vd)
            
            
            print('=============================')
            print('x :')
            print(x)
            
            print('pi :')
            print(pi)
            print('p :')
            print(p)   
            
            print('z :')
            print(z)
            print('v :')
            print(v)         
            
            loss = lMSE(z, v) + categorical_cross_entropy(p, pi)
            
            print('Loss: ', (lMSE(z, v).item(), categorical_cross_entropy(p, pi).item()))
            
            print('=============================')
            
            optimizer.zero_grad()  # (Re)Set all the gradients to zero
            loss.backward()  # Compute the backward pass based on the gradients and activations
            optimizer.step()  # Update the weights
            
            #print_network(network, optimizer)

            #metrics.update(y, t)
            
        '''
        metrics.compute("Train")
    
         # Validate the model
        print("Validating...")
        with torch.no_grad():
            metrics.reset()

            network.eval() # Freeze the network >< training mode
            for batch in val_loader:
                x, pi, z = batch
                #x, t = x.to(device), t.to(device)
    
                p, v = network(x)
                metrics.update(y, t)
            metrics.compute("Validation")
        print(metrics)
        metrics.clear()
        
        end = time.time()
        
        # Print logging
        print(f"\n-Ending epoch {epoch}: elapsed time {end - start}\n")
        '''
    #print('Network after training............................................')
    #print_network(network)
    network.eval()
    torch.save(network.state_dict(), new_model_path)

def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    y_true = torch.clamp(y_true, 1e-9, 1 - 1e-9)
    return (y_true * torch.log(y_true / y_pred)).sum(dim=1).mean()

def save_new_model(model_path, new_model_path):
    copyfile(new_model_path, model_path)

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
    def __init__(self, data):
        # # All the data preperation tasks can be defined here
        # - Deciding the dataset split (train/test/ validate)
        # - Data Transformation methods 
        # - Reading annotation files (CSV/XML etc.)
        # - Prepare the data to read by an index
        
        x = data[:,:11].astype(float)
        pi = data[:,11:16].astype(float)
        z = data[:,-1].astype(int)
        self.x = np.transpose(torch.from_numpy(x)).float()
        self.pi = np.transpose(torch.from_numpy(pi)).float()
        self.z = np.transpose(torch.from_numpy(z)).long().view(np.size(x,0),1)
        
         
    def __getitem__(self, index):
        # # Returns data and labels
        # - Apply initiated transformations for data
        # - Push data for GPU memory
        # - better to return the data points as dictionary/ tensor  
        return (self.x[:,index], self.pi[:,index], self.z[index])
 
    def __len__(self):
        return len(self.z)

def print_network(network):
    # Print model's state_dict
    mod_dict = network.state_dict()
    print("Model's state_dict:")
    for param_tensor in mod_dict:
        print(param_tensor, "\t", mod_dict[param_tensor].size())
        #print(mod_dict[param_tensor])
        print(torch.sum(mod_dict[param_tensor]))
    
    # Print optimizer's state_dict
    #opt_dict = optimizer.state_dict()
    #print("Optimizer's state_dict:")
    #for var_name in opt_dict:
    #    print(var_name, "\t", opt_dict[var_name])
        #print(opt_dict[var_name])   
        
