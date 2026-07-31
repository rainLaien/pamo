#!/usr/bin/env bash

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/222.stl \
    --output ./examples/test_outputs/222_remesh_only.stl \
    --remesh-only \
    --remesh-resolution 256 \
    --sdf-mode exact
