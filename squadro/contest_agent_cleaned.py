from agent import AlphaBetaAgent
from time import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import logging
import random

model_path = 'model/contest_agent3.pt'

"""
  Deep Network Class
"""
class DeepNetwork(nn.Module):
    def __init__(self):
        
        nin = 11             # 11 inputs: player id, 5 first numbers for the player 0 and five numbers for the player 1
        nout = 5             # 5 outputs: probability to choose one of the 5 actions
        hidden_layers = 200  # Size of the hidden layers

        self.batch_hid = nn.BatchNorm1d(num_features=hidden_layers)
        
        self.lin = nn.Linear(nin, hidden_layers)
        
        self.linp1 = nn.Linear(hidden_layers, hidden_layers)
        self.linp2 = nn.Linear(hidden_layers, hidden_layers)
        self.linp3 = nn.Linear(hidden_layers, nout)
        
        self.linv1 = nn.Linear(hidden_layers, hidden_layers)
        self.linv2 = nn.Linear(hidden_layers, hidden_layers)
        self.linv3 = nn.Linear(hidden_layers, 1)
        
     
    """
    Method called to evaluate the deep network
    """
    def forward(self, x):
        l = len(x.size())
        
        if l == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.batch_hid(self.lin(x)))
        
        ph = F.relu(self.batch_hid(self.linp1(x)))
        ph = F.relu(self.batch_hid(self.linp2(ph)))
        ph = self.linp3(ph)
        softph = F.softmax(ph, dim=-1)
        
        vh = F.relu(self.batch_hid(self.linv1(x)))
        vh = F.relu(self.batch_hid(self.linv2(vh)))
        vh = self.linv3(vh)
        
        return (softph, vh)

"""
Contest agent
"""
class MyAgent(AlphaBetaAgent):

    def __init__(self):
        self.action_size = 5         # Number of actions
        self.max_depth = 9           # Max depth of the MCTS (to remove)
        self.max_time = 0            # Max time for the simulations
        self.start_time = 0          # Start time of the simulation
        self.total_time = 0          # Total time of the game
        self.mcts = None
        self.MC_steps = 70           # Number of steps in MCTS
        self.turn_time = 0.03        # Percentage of total time allowed for each turn
        self.hurry_time = 0.2        # Percentage of total time when it begins to hurry up
        self.epsilonMove = 0.03      # Probability to choose randomly the move
        self.epsilonMCTS = 0.2       # Probability to choose randomly the node in MCTS (for TRAINING only)
        self.tau = 1                 # If MCTS stochastic: select action with distribution pi^(1/tau)
        
        self.results = None          # Store results
        
        self.deepnetwork = DeepNetwork()
        self.set_model_path(model_path)
        self.tensor_state = None


    def get_name(self):
        return 'Group 13'
    
    
    def set_model_path(self, path):
        self.deepnetwork.load_state_dict(torch.load(path))
        self.deepnetwork.eval()
    
    """
    Return the next action
    """
    def get_action(self, state, last_action, time_left):
        self.last_action = last_action
        self.time_left = time_left
        self.start_time = time()
        if self.total_time == 0:
            self.total_time = time_left;
        if time_left / self.total_time > self.hurry_time:
            self.max_time = self.turn_time * self.total_time
        else:
            self.max_time = self.turn_time * self.total_time * (time_left / (self.hurry_time * self.total_time))**2
        best_move = 1
        
        root = Node(state)
        self.mcts = MCTS(root)
        
        #### MCTS
        root_value = self.evaluateLeaf(root, 0)
        n = 1
        while time() - self.start_time < self.max_time: # TO REPLACE for contest
            self.simulate(state)
            n += 1
        
        pi, values = self.getAV()
        
        #### pick the action (stochastically with prob = epsilon)
        tau = self.tau if random.uniform(0, 1) < self.epsilonMove else 0
        best_move, value = self.chooseAction(pi, values, tau)

        nextState, _ = self.takeAction(state, best_move)

        NN_value = -self.evaluate(nextState)[1] # - sign because it evaluates with respect to the current player of the state
        
        return best_move
  
    """
    Simulate the tree search via Monte Carlo algorithm
    """
    def simulate(self, state):
      
        ##### MOVE THE LEAF NODE
        # Breadcrumbs = path from root to leaf
        leaf, done, breadcrumbs = self.mcts.moveToLeaf(self)
        
        ##### EVALUATE THE LEAF NODE with deep neural network + add edges to leaf node
        value = self.evaluateLeaf(leaf, done)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

  
    """
    Evaluate a node by means of the neural network
    """
    def evaluateLeaf(self, leaf, done):

        if done == 0:
    
            probs, value = self.evaluate(leaf.state)
            allowedActions = leaf.state.get_current_player_actions()

            probs = probs[allowedActions]

            # Add node in tree
            for idx, action in enumerate(allowedActions):
                newState, _ = self.takeAction(leaf.state, action)
                node = Node(newState)

                newEdge = Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))
                
        else: # End of game leaf
            value = 1 if leaf.state.cur_player == self.id else -1

        return value
    
    """
    Return the probability and values for each successor
    """
    def getAV(self):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        
        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1/self.tau)
            values[action] = edge.stats['Q']
        pi = pi / (np.sum(pi) * 1.0)
        return pi, values
      
    """
    Select the action in function of the computed probabilities
    """
    def chooseAction(self, pi, values, tau):
        if tau == 0: # Choose deterministically
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0] # if several states have the same prob
        else: # Choose stochastically (for TRAINING)
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0] # random action

        value = values[action]

        return action, value
        
      
    """
    Apply the action and return the new state
    """
    def takeAction(self, state, a):
      
        newState = state.copy()
        newState.apply_action(a)
        
        done = 1 if self.cutoff(newState, self.max_depth) else 0

        return (newState, done) 


    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """
    def successors(self, state):
        actions = state.get_current_player_actions()
        for a in actions:
            s = state.copy()
            s.apply_action(a)
            yield (a, s)


    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """
    def cutoff(self, state, depth):
        return state.game_over_check() or time() - self.start_time > self.max_time


    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """
    def evaluate(self, state):
        l1 = [state.get_pawn_advancement(state.cur_player, pawn) for pawn in [0, 1, 2, 3, 4]]
        l2 = [state.get_pawn_advancement(1 - state.cur_player, pawn) for pawn in [0, 1, 2, 3, 4]]
        x = torch.FloatTensor([state.cur_player] + l1 + l2)
        ph, vh = self.deepnetwork(x)
        ph = ph.data.numpy()[0,:]
        vh = np.float(vh.data.numpy())
        return (ph, vh) # Deep neural network evaluation


# Source: https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/
"""
Monte Carlo Tree Search class
"""
class MCTS():

    def __init__(self, root):
        self.root = root
        self.cpuct = 1
        
    """
    Move to a node not explored yet
    """
    def moveToLeaf(self, player):

        breadcrumbs = []
        currentNode = self.root

        done = 0

        while not currentNode.isLeaf():
            
            maxQU = -float('Inf')

            # Choose randomly at 20% for the root node (ONLY for the training)
            epsilon = player.epsilonMCTS if currentNode == self.root else 0
            nu = np.random.dirichlet([0.8] * len(currentNode.edges)) if currentNode == self.root else [0] * len(currentNode.edges)

            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):

                U = self.cpuct * np.sqrt(Nb) / (1 + edge.stats['N'])
                   
                Q = edge.stats['Q']

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge

            newState, done = player.takeAction(currentNode.state, simulationAction) # the value of the newState from the POV of the new playerTurn
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)

        return currentNode, done, breadcrumbs


    def backFill(self, leaf, value, breadcrumbs):

        currentPlayer = leaf.playerTurn

        for edge in breadcrumbs:
            playerTurn = edge.playerTurn
            direction = 1 if playerTurn == currentPlayer else -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']


"""
Node class
"""
class Node():

    def __init__(self, state):
        self.state = state
        self.playerTurn = state.get_cur_player()
        self.edges = []

    def isLeaf(self):
        return len(self.edges) == 0




"""
Edge class: it links two links between each other
"""
class Edge():

    def __init__(self, inNode, outNode, prior, action):
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.playerTurn
        self.action = action

        self.stats =  {
                    'N': 0,
                    'W': 0,
                    'Q': 0,
                    'P': prior,
                }
