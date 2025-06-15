#!/bin/bash

conda create -n test python=3.12 pip -y
conda activate test
pip install squadro
python -c "from squadro import game; game.run()"