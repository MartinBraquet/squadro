from squadro.tools.utils import AGENTS

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

### From PyPI

```
pip install squadro
```

### From source

```shell
git clone git@github.com:MartinBraquet/squadro.git
cd squadro
```

#### Environment

If not already done, create a virtual environment using your favorite environment manager. For instance using conda:

```shell
conda create -n squadro python=3.12
conda activate squadro
```

#### Prerequisites

If running on a Linux machine without intent to use a GPU, run this beforehand to install only the CPU version
of the `pytorch` library:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### Main Installation

Install the package in editable mode:

```shell
pip install -e .
```

## Usage

This package can be used in the following ways:

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

It will stop training when the evaluation loss stops improving. Once done, one can use the model; see the next
section below (setting the appropriate value for `model_path`, e.g., `'...'`).

### Play

#### Preliminaries

If you run into the following error when launching the game:
```
libGL error: failed to load driver
```

Then try setting the following environment variable beforehand:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

#### Play against another human

To play the game, run the following command:

```python
from squadro.animated_game import RealTimeAnimatedGame

RealTimeAnimatedGame(n_pawns=5, first=None).run()
```


To access all the parameters to play, see the doc:

```python
from squadro.animated_game import RealTimeAnimatedGame

help(RealTimeAnimatedGame.__init__)  # for the arguments to RealTimeAnimatedGame
```

#### Play against the computer

To play against the computer, set `agent_1` to a value in:

```python
from squadro.tools.utils import AVAILABLE_AGENTS

print(AVAILABLE_AGENTS)
```

For instance:

```python
from squadro.animated_game import RealTimeAnimatedGame

RealTimeAnimatedGame(n_pawns=5, first=None, agent_1='random').run()
```

#### Play against your trained AI

After training your AI as described in the [Training](#Training) section, you can play against her using:

```python
```

#### Play against a benchmarked AI

If you do not want to train a model, as described in the [Training](#Training) section, you can still play against
a benchmarked model available online. After passing `init_from='online'`, you can set `model_path` to any of those
currently supported models:

| `model_path` | # layers | # heads | embed dims | # params | size   |
|--------------|----------|---------|------------|----------|--------|
| `...`        | 12       | 12      | 768        | 124M     | 500 MB |

Note that the first time you use a model, it needs to be downloaded from the internet; so it can take a few minutes.

Example:

```python
...
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

For any issue / bug report / feature request,
open an [issue](https://github.com/MartinBraquet/squadro/issues).

## Contributions

To provide upgrades or fixes, open a [pull request](https://github.com/MartinBraquet/squadro/pulls).

### Contributors

[![Contributors](https://contrib.rocks/image?repo=MartinBraquet/squadro)](https://github.com/MartinBraquet/squadro/graphs/contributors)
