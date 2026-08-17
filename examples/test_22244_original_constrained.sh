#!/usr/bin/env bash
###
 # @FilePath: \pamo\examples\test_22244_original_constrained.sh
 # @Author: laien laien.rain@gmail.com
 # @Date: 2026-08-15 16:23:57
 # ----------------------------------------------
 # @LastEditors: laien laien.rain@gmail.com
 # @LastEditTime: 2026-08-15 16:30:30
 # @Description: 
 # ----------------------------------------------
 # Copyright (c) 2026 Supreium Co., Ltd , All Rights Reserved.
### 

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INPUT_PATH="${SCRIPT_DIR}/22244.stl"
OUTPUT_DIR="${SCRIPT_DIR}/test_outputs"
OUTPUT_PATH="${OUTPUT_DIR}/22244_constrained.stl"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
"${PYTHON_BIN}" -u ./example.py \
    --input "${INPUT_PATH}" \
    --output "${OUTPUT_PATH}" \
    --original-constrained-remesh \
    --sdf-mode exact \
    --constraint-feature-angle 15 \
    --coplanar-angle-tolerance 0.1 \
    --constraint-flip-passes 2 \
    --constraint-flip-minimum-valence 12 \
    --constraint-flip-maximum-candidate-quality 0.1 \
    --constraint-planar-fan-minimum-valence 30 \
    --constraint-planar-annulus-minimum-faces 20 \
    --constraint-cylinder-minimum-faces 20 \
    --constraint-cylinder-radius-tolerance 0.001 \
    --constraint-cylinder-target-edge-ratio 1.0 \
    --constraint-partial-cylinder-minimum-faces 20 \
    --constraint-partial-cylinder-radius-tolerance 0.002 \
    --constraint-partial-cylinder-normal-tolerance 0.02 \
    --constraint-partial-cylinder-minimum-angle 30 \
    --constraint-rounded-fillet-minimum-faces 12 \
    --constraint-rounded-fillet-minimum-curvature 0.2 \
    --constraint-planar-region-minimum-faces 20 \
    --constraint-quality-iterations 0 \
    --constraint-quality-step 0.4 \
    --constraint-quality-flip-passes 0

echo "Optimized mesh written to: ${OUTPUT_PATH}"
