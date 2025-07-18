[![Release](https://img.shields.io/pypi/v/squadro?label=Release&style=flat-square)](https://pypi.org/project/squadro/)
[![CI](https://github.com/MartinBraquet/squadro/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinBraquet/squadro/actions/workflows/ci.yml/badge.svg)
[![CD](https://github.com/MartinBraquet/squadro/actions/workflows/cd.yml/badge.svg)](https://github.com/MartinBraquet/squadro/actions/workflows/cd.yml/badge.svg)
[![Coverage](https://codecov.io/gh/MartinBraquet/squadro/branch/master/graph/badge.svg)](https://codecov.io/gh/MartinBraquet/squadro)
[![Downloads](https://static.pepy.tech/badge/squadro)](https://pepy.tech/project/squadro) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package proposes multiple features related to the Squadro game, whose most important one is an AI agent outperforming most other available algorithms, as well as average human players—namely, me.

## Documentation

Squadro is a two-player board game on a 5x5 board. The goal is to have four of our pawns perform a return trip before the opponent. Each pawn has a respective speed given by the number of dots (1–3) at their starting position. If an opponent's pawn crosses one of my pawns, then my pawn returns to the side of the board. 

Visit my [website](https://martinbraquet.com/research/#AI_Agent_for_Squadro_board_game) for a visual and qualitative description.

#### Demo

<img src="https://martinbraquet.com/wp-content/uploads/demo-1.gif" alt="drawing" width="300"/>

[Watch a full demo on YouTube](https://youtu.be/1KkTbFvQc1Y)

#### Other games?
The code is modular enough to be easily applied to other games. To do so, you must implement its state in [state.py](squadro/state/state.py), and make a few other changes in the code base depending on your needs. Please raise an issue if discussion is needed.

## Installation

The package works on any major OS (Linux, Windows, and MacOS) and with Python >= 3.11.

> [!TIP]
> If you have no intent to use a GPU, run this beforehand to install only the CPU version of the `pytorch` library (much lighter, and hence much faster to install):
> ```shell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```


The most straightforward way is to simply install it from PyPI via:
```bash
pip install squadro
```

If you want to install it from source, which is necessary for development, follow the instructions [here](docs/installation.md).

If some dependencies release changes that break the code, you can install the project from its lock file—which fixes the dependency versions to ensure reproducibility:
```bash
pip install -r requirements.txt
```

## Background

> [!NOTE]  
> This section is highly technical; feel free to skip it and play with the code right away.

#### Algorithms for Board Games

Most computer algorithms discretize the game into states and actions. Here, the state is the position of the pawns and the available actions are the possible moves of the pawns.

Squadro is a finite state machine, meaning that the next state of the game is completely determined by the current state and the action played. With this definition, one can see that the game is a Markov Decision Process (MDP). At each state, the current player can play different actions, which lead to different states. Then the next player can play different actions from any of those new states, etc. The future of the game can be represented as a tree, whose branches are the actions that lead to different states.

#### Exploration - Exploitation trade-off

An algorithm can explore that space of possibilities to infer the best move to play now. As the tree is huge, it is not possible to explore all the possible paths until the end of the game. Typically, they explore only a small fraction of the tree and then use the information gathered from those states to make a decision. More precisely, those two phases are:

* **State exploration**: exploring the space of states by a careful choice of actions. The most common exploration methods are Minimax and Monte Carlo Tree Search (MCTS). Minimax explores all the states up to a specific depth, while MCTS navigates until it finds a state that has not been visited yet. Minimax can be sped up by skipping the search in the parts of the tree that won't affect the final decision; this method is called alpha-beta pruning.
* **State evaluation**: evaluating a state. If we have a basic understanding of the game and how to win, one can design a heuristic (state evaluation function) that gives an estimate of how good it is to be in that state / position. Otherwise, it can often be better to use a computer algorithm to evaluate the state.
  * The simplest algorithm to estimate the state is to randomly let the game play until it is over (i.e., pick random actions for both players). When played enough times, it can give the probability to win in that state.
  * More complex, and hence accurate, algorithms are using reinforcement learning (AI). They learn from experience by storing information about each state/action in one of:
    * Q value function, a lookup table for each state and action;
    * deep Q network (DQN), a neural network that approximates the Q value function, which is necessary when the state space is huge (i.e., cannot be stored in memory).

#### Agents

At least 8 agents, each running a different algorithm, have been implemented to play the game:

* _human_: another local human player (i.e., both playing on the same computer)
* _random_: a computer that plays randomly among all available moves
* _ab_relative_advancement_: a computer that lists the possible moves from the current position and evaluates them directly (i.e., it "thinks" only one move ahead), where the evaluation function is the player's advancement
* _relative_advancement_: a computer that lists the possible moves from the current position and evaluates them directly (i.e., it "thinks" only one move ahead), where the evaluation function is the player's advancement compared to the other player
* _ab_relative_advancement_: a computer that plays minimax with alpha-beta pruning, where the evaluation function is the player's advancement compared to the other player
* _mcts_advancement_: Monte Carlo tree search, where the evaluation function is the player's advancement compared to the other player
* _mcts_rollout_: Monte Carlo tree search, where the evaluation function is determined by a random playout until the end of the game
* _mcts_q_learning_: Monte Carlo tree search, where the evaluation function is determined by a lookup table
* _mcts_deep_q_learning_: Monte Carlo tree search, where the evaluation function is determined by a convolutional neural network 

You can also access the most updated list of available agents with:

```python
import squadro

print(squadro.AVAILABLE_AGENTS)
```

#### Benchmark

All the agents have been evaluated against each other under controlled conditions:
- Max 3 sec per move
- 100 games per pairwise evaluation—exactly balanced across the four starting configurations—which color and who starts—except when a human is involved (only 5 games then)
- Original grid (5 x 5)
- Since all the top algorithms (MCTS and Minimax) are deterministic, we need to add small randomness to prevent each starting configuration from leading to the same game. We set `tau=.5, p_mix=0, a_dirichlet=0` to randomize the MCTS agents; the Minimax ones being les performant, we didn't bother with randomizing them for now.
- Hardware information:
	- CPU: AMD Ryzen 9 5900HX with Radeon Graphics (8 cores, 16 threads)
	- GPU: No NVIDIA GPU
	- RAM: 15.02 GB RAM
	- OS: Linux 6.11.0-29-generic (#29~24.04.1-Ubuntu SMP)
	- Python 3.12.9 (CPython)
	- Libraries: {'numpy': '2.0.1', 'torch': '2.6.0+cpu'}

See [benchmark.ipynb](notebooks/benchmark.ipynb) for code reproducibility.

Below is the pairwise algorithm comparison; the value for some row R and column C corresponds to the win rate of R against C. For example, I (human) beat MCTS deep Q-learning 20% of the time.

|                             | human | mcts deep q learning | mcts advancement | mcts rollout | ab relative advancement | relative advancement | advancement | random |
| :-------------------------- | ----: | -------------------: | ---------------: | -----------: | ----------------------: | -------------------: | ----------: | -----: |
| **human**                   |       |                  0.2 |              0.4 |            0 |                     0.8 |                    1 |           1 |      1 |
| **mcts deep q learning**    |   0.8 |                      |             0.75 |         0.24 |                    0.54 |                    1 |           1 |      1 |
| **mcts advancement**        |   0.6 |                 0.25 |                  |         0.06 |                    0.32 |                    1 |           1 |      1 |
| **mcts rollout**            |     1 |                 0.76 |             0.94 |              |                    0.77 |                 0.98 |        0.99 |      1 |
| **ab relative advancement** |   0.2 |                 0.46 |             0.68 |         0.23 |                         |                    1 |           1 |      1 |
| **relative advancement**    |     0 |                    0 |                0 |         0.02 |                       0 |                      |         0.5 |   0.97 |
| **advancement**             |     0 |                    0 |                0 |         0.01 |                       0 |                  0.5 |             |   0.95 |
| **random**                  |     0 |                    0 |                0 |            0 |                       0 |                 0.03 |        0.05 |        |


The MCTS rollout algorithm outperforms all other players, including the human (myself, an average player). The MCTS deep Q-learning (DQL) algorithm is second, although it beats MCTS rollout when allowed less than .2 second per move. 

There are two components determining the quality of a tree-search algorithm: the number of searches and the quality of evaluation at the end of each search. MCTS rollout has poorer evaluation quality but it makes vastly more (10x) searches than MCTS DQL. That's why MCTS rollout is more performant than MCTS DQL (at least when the neural-network inference runs slowly on a CPU—instead of a GPU).

The question whether a specific type of MCTS DQL may ever beat MCTS rollout at Squadro is still left hanging. It is well-known that RL algorithms outperform more basic tree-search algorithms (like rollout) for games like Chess and Go. But those games have a much larger state space than Squadro 5x5. For small state spaces, heavy state evaluation through neural networks tends to have less value since rolling out from one state to the end is very quick.

## Usage

This package can be used in many interesting ways. You can play the game, train an AI agent, run simulations, and analyze animations.

### Play

You can play against someone else or many different types of computer algorithms.

> [!TIP]
> If you run into the following error on a Linux machine when launching the game:
> > libGL error: failed to load driver
> 
> Then try setting the following environment variable beforehand:
> ```
> export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
> ```

> [!TIP]
> To see detailed information about how the program is running, you can set up a logger by adding this at the start of your code:
> ```python
> import squadro
> squadro.logger.setup()
> ```

#### Play against another human

To play the game with someone else, run the following command:

```python
import squadro

squadro.GamePlay(n_pawns=5, first=None).run()
```

To access all the parameters to play, see the doc:

```python
help(squadro.GamePlay.__init__)  # for the arguments to RealTimeAnimatedGame
```

#### Play against the computer

To play against the computer, set `agent_1` to one of the `squadro.AVAILABLE_AGENTS`.

For instance:

```python
squadro.GamePlay(agent_1='random').run()
```

> [!TIP]
> To play against our best algorithm, run:
> ```python
> squadro.GamePlay(agent_1='best').run()
> ```
> Let us know if you ever beat it!

#### Play against your trained AI

After training your AI (see [Training](#Training) section), you will be able to play against her.

#### Play against an online pre-trained AI


If you do not want to train a model, as described in the [Training](#Training) section, you can still play against a model that the developers have already trained for you. They are available on [Hugging Face](https://huggingface.co/martin-shark/squadro/tree/main)—no need to download them from the browser, as they will automatically be downloaded once you run the code below.

Here are the online pre-trained models:

| Agent      | # pawns | size   |
| ---------- | ------- | ------ |
| Q-Learning | 2       | 18 kB  |
| Q-Learning | 3       | 6.2 MB |

| Agent           | # pawns | # CNN layers | # res blocks | # params | size   |
| --------------- | ------- | ------------ | ------------ | -------- | ------ |
| Deep Q-Learning | 3       | 64           | 4            | 380 k    | 1.5 MB |
| Deep Q-Learning | 4       | 128          | 6            | 1.8 M    | 7.1 MB |
| Deep Q-Learning | 5       | 128          | 6            | 1.8 M    | 7.1 MB |

Those models are all very lightweight, making them convenient even for machines with limited resources and fast games.

To use those models, simply instantiate the corresponding agent **without** passing the `model_path` argument (this is how the package makes the distinction between loading an online model and creating a new model).

```python
dql = squadro.MonteCarloDeepQLearningAgent() # Deep Q-Learning Opponent
squadro.GamePlay(agent_1=dql).run()
```

```python
ql = squadro.MonteCarloQLearningAgent() # Q-Learning Opponent
squadro.GamePlay(agent_1=ql, n_pawns=3).run()
```

### Training

One can train a model using reinforcement learning (RL) algorithms. Currently, Squadro supports two such algorithms:

#### Q-Learning

One needs to train a lookup table mapping each state to its value.

```python
import squadro

squadro.logger.setup(section='training')

trainer = squadro.QLearningTrainer(
    n_pawns=3,
    lr=.3,
    eval_steps=100,
    eval_interval=300,
    n_steps=100_000,
    parallel=8,
    model_path='path/to/model'
)
trainer.run()
```

For a 3x3 grid, it should take a few hours to train on a typical CPU (8–16 cores). For larger grids like the original 5x5 ones, the number of states is most likely too large to be stored in memory or computed by your computer in the foreseeable future.

You may however be able to adjust the code so that it does not load the entire table in memory. In such case, you would train the agent by building an ever-growing table in disk.

Note that there are many more parameters to tweak, if desired. See all of them in the doc:

```python
help(squadro.QLearningTrainer)
```

#### Deep Q-Learning

Here, the state-action value is approximated by a neural network.

```python
import squadro

squadro.logger.setup(section=['training', 'benchmark'])

trainer = squadro.DeepQLearningTrainer(
    eval_games=50,
    eval_interval=300,
    backprop_interval=20,
    model_path='path/to/model',
    model_config=squadro.ModelConfig(),
    init_from=None,
    n_pawns=5,
)
trainer.run()
```

For three pawns, it should take a few hours to train on a typical CPU (8–16 cores), and it is much faster on a GPU. For five pawns, it may take a few days.

Below is an example of good training metrics.
- The self-play win rate stays around 50%.
- The replay buffer samples remain diverse (above 80%).
- The policy and value losses slowly decrease.
- The win rate against its checkpoint is above 50% (checkpoint is replaced by the current model when the win rate goes above 70%)
- The elo is smoothly increasing.

Note that in reinforcement learning, the loss is not a key metrics to measure model improvement, as the training samples are constantly improving. Better metrics include the win rate against its checkpoint and elo. Once the latter metrics stabilize, the model has reached its peak.

![](https://martinbraquet.com/wp-content/uploads/training_plots.png)

Once done, one can play against the AI agent (setting the same value for `model_path`):

```python
agent = squadro.MonteCarloDeepQLearningAgent(
    model_path='path/to/model',
    max_time_per_move=1.,
)
squadro.GamePlay(agent_1=agent).run()
```

### Simulations

You can simulate a game between two computer algorithms. Set `agent_0` and `agent_1` to any of the `AVAILABLE_AGENTS` above and run:

```python
game = squadro.Game(agent_0='random', agent_1='random')
game.run()
print(game)
game.to_file('game_results.json')
```

### Animations

You can render an animation of any game. Press the left and right keys to navigate through the game.

To get the game you want to animate, you can either simulate a new one.

```python
game = squadro.Game(agent_0='random', agent_1='random')
squadro.GameAnimation(game).show()
```

Or you can load the game from a file.

```python
game = squadro.Game.from_file('game_results.json')  
print(game.to_dict())  
squadro.GameAnimation(game).show()
```

## Tests

```shell
pytest squadro
```

## Feedback

For any issue / bug report / feature request, open an [issue](https://github.com/MartinBraquet/squadro/issues).

## Contributions

To provide upgrades or fixes, open a [pull request](https://github.com/MartinBraquet/squadro/pulls).

### Contributors

[![Contributors](https://contrib.rocks/image?repo=MartinBraquet/squadro)](https://github.com/MartinBraquet/squadro/graphs/contributors)
