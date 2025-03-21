#!/bin/bash

# Provides inputs/outputs pairs (chosen by the smart agent, which are quite good)
# The random agent is useful to have plenty of different states
# Inputs = states: 5 positions of each player (list of 10 elements)
# Outputs = actions: 5 possible actions for player 0

while true
do
   python3 squadro_no_GUI.py -ai0 smart_agent_for_training -ai1 random_agent -f 1
done

