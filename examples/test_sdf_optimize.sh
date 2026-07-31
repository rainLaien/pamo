#!/usr/bin/env bash
###
 # @FilePath: \pamo\examples\test_sdf_optimize.sh
 # @Author: laien laien.rain@gmail.com
 # @Date: 2026-07-31 11:04:55
 # ----------------------------------------------
 # @LastEditors: laien laien.rain@gmail.com
 # @LastEditTime: 2026-07-31 11:28:55
 # @Description: 
 # ----------------------------------------------
 # Copyright (c) 2026 Supreium Co., Ltd , All Rights Reserved.
### 

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/222.stl \
    --output ./examples/test_outputs/222_sdf_optimized.stl \
    --sdf-optimize \
    --remesh-resolution 256 \
    --sdf-mode exact \
    --sdf-optimize-iterations 5 \
    --sdf-smoothing-step 0.2 \
    --sdf-projection-steps 3 \
    --sdf-feature-angle 45
