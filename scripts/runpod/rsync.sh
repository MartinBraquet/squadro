#!/bin/bash

set -e

IP=157.157.221.30

folders=(
  "martin"
  "squadro"
  "scripts/runpod"
  "pyproject.toml"
)

for folder in "${folders[@]}"; do
  rsync -rvz -e 'ssh -o StrictHostKeyChecking=no -p 30939 -i ~/.ssh/runpod' --progress --exclude '*.pt' --exclude '*.json' --exclude '__pycache__*' ../../$folder root@$IP:/workspace
done
