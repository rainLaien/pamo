#!/usr/bin/env bash

set -e

bash ./examples/test_remesh_only.sh
bash ./examples/test_feature_remesh.sh
bash ./examples/test_feature_optimize.sh
bash ./examples/test_surface_sample_remesh.sh
bash ./examples/test_sdf_optimize.sh
bash ./examples/test_original_constrained.sh
bash ./examples/test_full_pipeline.sh
