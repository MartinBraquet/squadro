# Squadro


[![Release](https://img.shields.io/pypi/v/squadro?label=Release&style=flat-square)](https://pypi.org/project/squadro/)
[![CI](https://github.com/MartinBraquet/squadro/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinBraquet/squadro/actions/workflows/ci.yml/badge.svg)
[![CD](https://github.com/MartinBraquet/squadro/actions/workflows/cd.yml/badge.svg)](https://github.com/MartinBraquet/squadro/actions/workflows/cd.yml/badge.svg)
[![Coverage](https://codecov.io/gh/MartinBraquet/squadro/branch/main/graph/badge.svg)](https://codecov.io/gh/MartinBraquet/squadro)
[![Documentation Status](https://readthedocs.org/projects/squadro/badge/?version=latest)](https://squadro.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/squadro)](https://pepy.tech/project/squadro) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository: https://github.com/MartinBraquet/squadro.

![Alt Text](https://raw.githubusercontent.com/MartinBraquet/squadro/img/demo.gif)

## Installation from PyPI

```
pip install squadro
```

If you run into the following error when running the game:
```
libGL error: failed to load driver
```

Then try setting the following environment variable beforehand:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

### Usage

To play the game, run the following command:
```
python -m squadro
```


## Installation from Source

```
pip install -r requirements.txt
```

## Documentation

Click [here](https://martinbraquet.com/index.php/research/#Squadro) for a full description.

Visualization of the convolutional neural network:

```
nn_visualization.ipynb
```

![](https://raw.githubusercontent.com/MartinBraquet/squadro/main/src/squadro/nn1.png)

![](https://raw.githubusercontent.com/MartinBraquet/squadro/main/src/squadro/nn2.png)

## Training

Train the model and save it as `model.pt`.

```
squadro_training.ipynb
```

Accuracy vs epochs.

![](https://raw.githubusercontent.com/MartinBraquet/squadro/main/src/squadro/accuracy.png)

Loss vs epochs.

![](https://raw.githubusercontent.com/MartinBraquet/squadro/main/src/squadro/loss.png)

## Test

Test in Jupiter Notebook. The model can be loaded from the training above in `model.pt` or from the 
default precise model in `model_precise.pt`.

```
squadro_test.ipynb
```

Test in Python.

```
python src/squadro/drawing.py
```

## Tools

Draw a digit and save it as a PNG file.

```
user_input_drawing.ipynb
```

## Issues / Bug reports / Feature requests

Please open an issue.

## Contributions

Contributions are welcome. Please check the outstanding issues and feel free to open a pull request.

### Contributors

[![Contributors](https://contrib.rocks/image?repo=MartinBraquet/squadro)](https://github.com/MartinBraquet/squadro/graphs/contributors)
