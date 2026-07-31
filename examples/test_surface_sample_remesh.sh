#!/usr/bin/env bash

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/222.stl \
    --output ./examples/test_outputs/222_surface_sample_remeshed.stl \
    --surface-sample-remesh \
    --surface-sample-count 5000 \
    --surface-sample-oversample 4 \
    --surface-sample-seed 0 \
    --feature-edge-angle 30 \
    --surface-flip-passes 32 \
    --surface-max-edge-ratio 2.0 \
    --surface-min-edge-ratio 0.5 \
    --surface-split-passes 64 \
    --surface-collapse-passes 24 \
    --surface-relax-iterations 2 \
    --surface-relax-step 0.2 \
    --surface-barycentric-margin 0.08
