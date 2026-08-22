#!/usr/bin/env bash
###
 # @FilePath: \pamo\examples\test_222_thinwall.sh
 # @Author: laien laien.rain@gmail.com
 # @Date: 2026-08-15 16:42:24
 # ----------------------------------------------
 # @LastEditors: laien laien.rain@gmail.com
 # @LastEditTime: 2026-08-15 16:46:04
 # @Description: 
 # ----------------------------------------------
 # Copyright (c) 2026 Supreium Co., Ltd , All Rights Reserved.
### 

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INPUT_PATH="${SCRIPT_DIR}/222_li.stl"
OUTPUT_DIR="${SCRIPT_DIR}/test_outputs"
OUTPUT_PATH="${OUTPUT_DIR}/222_primary_planes_rebuilt.stl"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "Input mesh not found: ${INPUT_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

# Classify the complete surface, then rebuild only the largest opposite-facing
# planar pair (the primary top and bottom skins). Their old internal edges are
# discarded. Circular/cylindrical, fillet, extrusion and remaining curved
# regions retain their original surface geometry and connectivity; only long
# shared boundary edges receive conforming collinear samples.
"${PYTHON_BIN}" -u ./example.py \
    --input "${INPUT_PATH}" \
    --output "${OUTPUT_PATH}" \
    --original-constrained-remesh \
    --constraint-allow-open-surface \
    --sdf-mode exact \
    --constraint-feature-angle 15 \
    --constraint-max-edge-length 1000 \
    --coplanar-angle-tolerance 0.5 \
    --coplanar-distance-ratio 0.00001 \
    --constraint-flip-passes 0 \
    --constraint-planar-annulus-minimum-faces 20 \
    --constraint-planar-region-minimum-faces 20 \
    --constraint-planar-largest-opposed-pair-only \
    --constraint-planar-target-edge-length 10 \
    --constraint-planar-minimum-angle 28 \
    --constraint-quality-iterations 0 \
    --constraint-quality-step 0.4 \
    --constraint-quality-flip-passes 0

echo "Primary top/bottom plane rebuild written to: ${OUTPUT_PATH}"
