#!/bin/bash

conda create -n test python=3.10 pip -y
conda activate test
pip install squadro
python -c "from squadro import drawing; drawing.run()"