#!/usr/bin/env bash

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/222.stl \
    --output ./examples/test_outputs/222_feature_optimized.stl \
    --feature-optimize \
    --remesh-resolution 128 \
    --sdf-mode exact \
    --projection-iterations 2 \
    --feature-edge-angle 30 \
    --feature-edge-target-length 8 \
    --sdf-optimize-iterations 5 \
    --sdf-smoothing-step 0.2 \
    --sdf-projection-steps 3 \
    --feature-quality-iterations 3 \
    --feature-quality-step 0.2 \
    --feature-flip-passes 1
