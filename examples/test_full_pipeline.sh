#!/usr/bin/env bash

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/222.stl \
    --output ./examples/test_outputs/222_full_pipeline.stl \
    --ratio 0.1 \
    --min-vertex 0 \
    --sdf-mode exact
