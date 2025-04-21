# Squadro


[![Release](https://img.shields.io/pypi/v/squadro?label=Release&style=flat-square)](https://pypi.org/project/squadro/)
[![CI](https://github.com/MartinBraquet/squadro/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinBraquet/squadro/actions/workflows/ci.yml/badge.svg)
[![CD](https://github.com/MartinBraquet/squadro/actions/workflows/cd.yml/badge.svg)](https://github.com/MartinBraquet/squadro/actions/workflows/cd.yml/badge.svg)
[![Coverage](https://codecov.io/gh/MartinBraquet/squadro/branch/main/graph/badge.svg)](https://codecov.io/gh/MartinBraquet/squadro)
[![Documentation Status](https://readthedocs.org/projects/squadro/badge/?version=latest)](https://squadro.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/squadro)](https://pepy.tech/project/squadro) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Alt Text](https://raw.githubusercontent.com/MartinBraquet/squadro/img/demo.gif)


## Documentation

Click [here](https://martinbraquet.com/index.php/research/#Squadro) for a full description.


## Demo

...

## Installation

> [!TIP]
> If running on a Linux machine without intent to use a GPU, run this beforehand to install only the CPU version of the `pytorch` library:
> ```shell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```


The most straightforward way is to simply install it from PyPI via:
```
pip install squadro
```

If you want to install it from source, which is necessary for development, follow the instructions [here](docs/installation.md).

## Usage

This package can be used in the following ways:

### Play

You can play against someone else or many different types of computer algorithms. See the [Agents](#Agents) section below for more details.

> [!TIP]
> If you run into the following error when launching the game:
> > libGL error: failed to load driver
> 
> Then try setting the following environment variable beforehand:
> ```
> export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
> ```

#### Play against another human

To play the game with someone else, run the following command:

```python
squadro.RealTimeAnimatedGame(n_pawns=5, first=None).run()
```

To access all the parameters to play, see the doc:

```python
help(squadro.RealTimeAnimatedGame.__init__)  # for the arguments to RealTimeAnimatedGame
```

#### Play against the computer

To play against the computer, set `agent_1` to one of the `squadro.AVAILABLE_AGENTS`.

For instance:

```python
squadro.RealTimeAnimatedGame(n_pawns=5, first=None, agent_1='random').run()
```

> [!TIP]
> To play against our best algorithm, run:
> ```python
> squadro.RealTimeAnimatedGame(agent_1='best').run()
> ```
> Let us know if you ever beat it!

#### Play against your trained AI

After training your AI as described in the [Training](#Training) section, you can play against her using:

```python
```

#### Play against a benchmarked AI

If you do not want to train a model, as described in the [Training](#Training) section, you can still play against a benchmarked model available online. After passing `init_from='online'`, you can set `model_path` to any of those currently supported models:

| `model_path` | # layers | # heads | embed dims | # params | size   |
|--------------|----------|---------|------------|----------|--------|
| `...`        | 12       | 12      | 768        | 124M     | 500 MB |

Note that the first time you use a model, it needs to be downloaded from the internet; so it can take a few minutes.

Example:

```python
...
```

#### Agents

Most computer algorithms discretize the game into states and actions. Here, the state is the position of the pawns and the available actions are the possible moves of the pawns.

Squadro is a finite state machine, meaning that the next state of the game is completely determined by the current state and the action played. With this definition, one can see that the game is a Markov Decision Process (MDP). At each state, the current player can play different actions, which lead to different states. Then the next player can play different actions from any of those new states, etc. The future of the game can be represented as a tree, whose branches are the actions that lead to different states.

An algorithm can explore that space of possibilities to infer the best move to play now. As the tree is huge, it is not possible to explore all the possible paths until the end of the game. Typically, they explore only a small fraction of the tree and then use the information gathered from those states to make a decision. More precisely, those two phases are:

* **State exploration**: exploring the space of states by a careful choice of actions. The most common exploration methods are Minimax and Monte Carlo Tree Search (MCTS). Minimax explores all the states up to a specific depth, while MCTS navigates until it finds a state that has not been visited yet. Minimax can be sped up by skipping the search in the parts of the tree that won't affect the final decision; this method is called alpha-beta pruning.
* **State evaluation**: evaluating a state. If we have a basic understanding of the game and how to win, one can design a heuristic (state evaluation function) that gives an estimate of how good it is to be in that state / position. Otherwise, it can often be better to use a computer algorithm to evaluate the state.
  * The simplest algorithm to estimate the state is to randomly let the game play until it is over (i.e., pick random actions for both players). When played enough times, it can give the probability to win in that state.
  * More complex, and hence accurate, algorithms are using reinforcement learning (AI). They learn from experience by storing information about each state/action in one of:
    * Q value function, a lookup table for each state and action;
    * deep Q network (DQN), a neural network that approximates the Q value function, which is necessary when the state space is huge (i.e., cannot be stored in memory).

List of available agents:

* _human_: another local human player (i.e., both playing on the same computer)
* _random_: a computer that plays randomly among all available moves
* _ab_advancement_: a computer that lists the possible moves from the current position and evaluates them directly (i.e., it "thinks" only one move ahead), where the evaluation function is the player's advancement
* _ab_relative_advancement_: a computer that lists the possible moves from the current position and evaluates them directly (i.e., it "thinks" only one move ahead), where the evaluation function is the player's advancement compared to the other player
* _ab_advancement_deep_: a computer that plays minimax with alpha-beta pruning (depth ~4), where the evaluation function is the player's advancement compared to the other player
* _mcts_advancement_: Monte Carlo tree search, where the evaluation function is the player's advancement compared to the other player
* _mcts_rollout_: Monte Carlo tree search, where the evaluation function is determined by a random playout until the end of the game

You can also access the most updated list of available agents with:

```python
import squadro

print(squadro.AVAILABLE_AGENTS)
```


### Training

One can train a model from scratch via:

```python
from squadro.train import Trainer

trainer = Trainer(
    model_path='results/tolstoy',  # output directory where the model will be saved
    training_data_path='https://www.gutenberg.org/cache/epub/2600/pg2600.txt',  # dataset URL or local path
    eval_interval=10,  # when to evaluate the model
    batch_size=4,  # batch size
    block_size=16,  # block size (aka context length)
    n_layer=2,  # number of layers
    n_head=4,  # number of attention heads per layer
    n_embd=32,  # embedding dimension
    dropout=0.2,  # dropout rate
    learning_rate=0.05,  # learning rate
    min_lr=0.005,  # minimum learning rate
    beta2=0.99,  # adam beta2 (should be reduced for larger models / datasets)
)
trainer.run()
```

It should take a few minutes to train on a typical CPU (8-16 cores), and it is much faster on a GPU.

Note that there are many more parameters to tweak, if desired. See all of them in the doc:

```python
help(Trainer)
```

It will stop training when the evaluation loss stops improving. Once done, one can use the model; see the next section below (setting the appropriate value for `model_path`, e.g., `'...'`).

### Simulations

You can simulate a game between two computer algorithms. Set `agent_0` and `agent_1` to any of the `AVAILABLE_AGENTS` above and run:

```python
game = squadro.Game(agent_0='basic', agent_1='random')
game.run()
print(game)
game.save_results('game_results.json')
```

### Animations

You can render an animation of a game between two computer algorithms. Press the left and right keys to navigate through the game.

```python
game = squadro.Game(agent_0='basic', agent_1='random')
squadro.GameAnimation(game).show()
```

### Profiling

You can also profile (memory, CPU and GPU usage, etc.) and benchmark the training process via:

```python
...(
    profile=True,
    profile_dir='profile_logs',
    ...
)
```

Then you can launch tensorboard and open http://localhost:6006 in your browser to watch in real time (or after hand) the training process.

```shell
tensorboard --logdir=profile_logs
```

### User Interface

...

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
