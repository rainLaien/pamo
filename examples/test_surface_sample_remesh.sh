#!/usr/bin/env bash
###
 # @FilePath: \pamo\examples\test_surface_sample_remesh.sh
 # @Author: laien laien.rain@gmail.com
 # @Date: 2026-07-31 14:07:34
 # ----------------------------------------------
 # @LastEditors: laien laien.rain@gmail.com
 # @LastEditTime: 2026-07-31 18:45:49
 # @Description:
 # ----------------------------------------------
 # Copyright (c) 2026 Supreium Co., Ltd , All Rights Reserved.
###

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/Unnamed-BodyPad.stl \
    --output ./examples/test_outputs/Unnamed-BodyPad_surface_sample_remeshed.stl \
    --surface-sample-remesh \
    --surface-sample-count 5000 \
    --surface-sample-oversample 4 \
    --surface-sample-seed 0 \
    --feature-edge-angle 20 \
    --surface-flip-passes 16 \
    --surface-max-edge-ratio 2.0 \
    --surface-min-edge-ratio 0.5 \
    --surface-split-passes 64 \
    --surface-collapse-passes 128  \
    --surface-protect-source-quality 0.95 \
    --surface-max-normal-deviation 5 \
    --surface-max-deviation-ratio 0.05 \
    --surface-min-collapse-quality 0.5 \
    --surface-coplanar-angle 1.0 \
    --surface-relax-iterations 10 \
    --surface-relax-step 0.2 \
    --surface-barycentric-margin 0.08
