#!/usr/bin/env bash
###
 # @FilePath: \pamo\examples\test_original_constrained.sh
 # @Author: laien laien.rain@gmail.com
 # @Date: 2026-07-31 11:05:00
 # ----------------------------------------------
 # @LastEditors: laien laien.rain@gmail.com
 # @LastEditTime: 2026-07-31 11:54:58
 # @Description: 
 # ----------------------------------------------
 # Copyright (c) 2026 Supreium Co., Ltd , All Rights Reserved.
### 

set -e

mkdir -p ./examples/test_outputs

python ./example.py \
    --input ./examples/222.stl \
    --output ./examples/test_outputs/222_original_constrained.stl \
    --original-constrained-remesh \
    --sdf-mode exact \
    --constraint-max-edge-length 10 \
    --constraint-feature-angle 5 \
    --constraint-max-splits 1000
