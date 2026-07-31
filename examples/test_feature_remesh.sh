#!/usr/bin/env bash
###
 # @FilePath: \pamo\examples\test_feature_remesh.sh
 # @Author: laien laien.rain@gmail.com
 # @Date: 2026-07-31 11:04:51
 # ----------------------------------------------
 # @LastEditors: laien laien.rain@gmail.com
 # @LastEditTime: 2026-07-31 11:13:32
 # @Description: 
 # ----------------------------------------------
 # Copyright (c) 2026 Supreium Co., Ltd , All Rights Reserved.
### 

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/222.stl \
    --output ./examples/test_outputs/222_feature_remesh.stl \
    --feature-remesh \
    --remesh-resolution 256 \
    --sdf-mode exact \
    --projection-iterations 2 \
    --feature-edge-angle 5 \
    --feature-edge-target-length 10
