"""
Advantages compared to minimax:
MCTS does not necessarily require a heuristic as it can do random playouts, good when no knowledge of the game
MCTS accuracy increases smoothly with computing time, since every node expansion slightly modifies the decision weights.
Minimax accuracy is more like a step function
due to iterative deepening. It makes no update on the decision weights (i.e., action to take) until it finishes exploring the
tree all the way down to depth k, which takes many node expansions.

Most of the time for MCTS is spent on the random playouts (function to compute next state from random action)
Most of the time for Minimax is spent on state copies, keeping them in memory, and their evaluation
"""
