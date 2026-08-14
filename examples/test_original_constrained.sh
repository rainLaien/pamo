#!/usr/bin/env bash
###
 # @FilePath: \pamo\examples\test_original_constrained.sh
 # @Author: laien laien.rain@gmail.com
 # @Date: 2026-07-31 11:05:00
 # ----------------------------------------------
 # @LastEditors: laien laien.rain@gmail.com
 # @LastEditTime: 2026-08-14
 # @Description: Refine Unnamed-Body.stl while preserving original features.
 # ----------------------------------------------
 # Copyright (c) 2026 Supreium Co., Ltd , All Rights Reserved.
###

set -e

mkdir -p ./examples/test_outputs

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ./.venv/Scripts/python.exe ]]; then
    PYTHON_BIN=./.venv/Scripts/python.exe
fi

"${PYTHON_BIN}" ./example.py \
    --input ./examples/Unnamed-Body.stl \
    --output ./examples/test_outputs/Unnamed-Body_original_constrained.stl \
    --original-constrained-remesh \
    --sdf-mode exact \
    --constraint-max-edge-length 10 \
    --constraint-feature-angle 5 \
    --constraint-max-splits 5000 \
    --coplanar-angle-tolerance 0.1 \
    --constraint-flip-passes 8 \
    --constraint-quality-iterations 20 \
    --constraint-quality-step 0.4 \
    --constraint-quality-flip-passes 12
