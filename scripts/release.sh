#!/bin/bash

cd "$(dirname "$0")"/..

set -e

RELEASE=$1
if [ -z "$RELEASE" ]
then
    RELEASE=0
fi

rm -rf dist
python -m build
if [ "$RELEASE" -eq 0 ]; then
    twine check dist/*
else
    twine upload dist/*
fi