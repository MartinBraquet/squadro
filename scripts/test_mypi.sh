#!/bin/bash

conda create -n test python=3.12 pip -y
conda activate test
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install squadro
python -c "from squadro import Game; Game().run()"