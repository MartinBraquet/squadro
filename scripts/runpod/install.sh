#!/bin/bash

set -e

IP=56fyshrt7za9gs-6441182f@ssh.runpod.io

ssh -tt $IP -i ~/.ssh/runpod << EOF
  set -e
  apt update
  apt install -y rsync
  exit
EOF

cd $(dirname "$0")
./rsync.sh

ssh -tt $IP -i ~/.ssh/runpod << EOF
  cd workspace
  sudo apt update && sudo apt upgrade -y
  sudo apt install python3.12 -y
  sudo add-apt-repository ppa:deadsnakes/ppa -y
  sudo apt update
  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2
  sudo update-alternatives --config python3
  curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.12
  pip install -e .
  exit
EOF