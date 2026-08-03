#!/usr/bin/env bash

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/Unnamed-BodyPad.stl \
    --output ./examples/test_outputs/Unnamed-BodyPad_surface_sample_remeshed_arc.stl \
    --surface-sample-remesh \
    --surface-protect-source-quality 0.8 \
    --surface-max-normal-deviation 5 \
    --surface-max-deviation-ratio 0.05