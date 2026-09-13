# Cone-only diagnostic partition

Latest update: classification growth now defaults to 0.02 input units.
Post-classification end transitions are exported separately as Freeform or
certified Torus candidates with feature_role=2. Thus this diagnostic can now
contain transition types in addition to Cone and Unknown. See END_TRANSITIONS.md;
the historical 0.8 policy below is no longer the default.

Current distance policy: Cylinder/Cone core certification and discovery use
at most one normalized fitting tolerance (or a stricter user fit ratio).
Post-core absorption and small-strip reassignment use a fixed absolute
ModelRuledGrowthDistance, default 0.8 input units. CLI override:
`--ruled-growth-distance 0.8`. Input in millimeters therefore means 0.8 mm;
STL itself carries no unit metadata. The growth distance no longer increases
with the fitted patch's error. Core and growth normal checks remain separate.
This supersedes the previous measured-error growth envelope for these types.
Accepted expanded patches can exceed the strict core fitting error because
their added faces use the explicitly requested absolute expansion tolerance.

Run `examples/partition_cones_only.ps1` with InputStl, OutputDirectory and
AnalyticSeedBackend (cpu/cuda/auto), or use the native `--cones-only` option.
The executable must be rebuilt separately. The script does not build it.

This mode follows the cylinder-only diagnostic pipeline but proposes and
refits only cones. CUDA prefetch also requests only cone fits. Sampling cells
do not assign planes. Core recognition retains its existing thresholds;
ring absorption uses the existing 30-degree normal limit and fixed model
distance tolerance. Remaining connected faces are Unknown, not Freeform.

The output contains primitive_type=3 for cones and primitive_type=0 for
unrecognized regions, saved in patch_result.ply and patch_report.json.
All faces are retained; no remesh or other primitive classification is run.
The option automatically enables partition-only snapshot export, and cannot
be combined with cylinders-only, legacy, remesh or saved-partition input.
Default full partition and the cylinder-only option remain available.

Existing candidate, adjacency and growth limitations still apply. No
compilation or runtime validation was performed for this change.

Cone-only discovery now includes a bounded cross-strip proposal pass before
the existing searches: at most 64 spatial-index-distributed slender-face seeds,
12 adjacency layers and 256 faces per seed, with fits at intermediate layers.
It crosses long-side adjacency without equal-length requirements, preserves
hard edges and restricts face normals to 60 degrees from the seed (unsigned).
This topological sweep proposes circumferential support; it is not guaranteed
to follow the true circumference before a cone axis is known. Proposals still
pass the existing geometric fit and area-support checks.

Before accepting or committing a cone core, centroid radial directions must
cover at least 30 degrees about its fitted axis. Each third of the occupied
angular interval must contribute at least 5 percent of the support area. The
interval is cut at its largest empty gap, avoiding angular-seam dependence.
This rejects poorly observed narrow cores; it does not prove parameter
uniqueness and can leave genuine very narrow cone fragments unrecognized.
The cone fitter already initializes the apex by intersecting tangent planes;
this change does not add an independent generator-line apex solver.

Certified cone growth may cross soft discovery bottlenecks between the long
sides of slender triangles, still requiring the original model compatibility.
The core normal threshold and final 30-degree absorption threshold are
unchanged. These additions are scoped to cones-only mode. Logs report new
proposal attempts/acceptance and angular-support rejection counts.

Cone-only ring absorption no longer stops at twelve face layers. It continues
until no additions, an empty frontier or the existing 100,000 evaluation cap.
Other modes retain their round limit. Rejection counters distinguish vertex
distance, normals and unsupported/unreliable samples (evaluation counts, not
unique faces). At most eight distance rejection examples include face/patch
ids and actual normalized error/limit. A single final edge scan reports
accessible versus adjacency-blocked cone/residual interfaces. These are
diagnostics, not evidence that every residual should belong to the cone.

After initial absorption, cone-only mode now runs the spatial residual reserve
with ModelResidualSeedBudget (default 1250), then absorbs again. Residual
neighborhood proposals and CUDA prefetch fit only cones; mixed whole-component
model selection is skipped in this mode. The angular-support certificate and
normal/distance criteria are preserved. Full-mode behavior is unchanged.

`python examples/diagnose_cone_residuals.py OUTPUT_DIRECTORY` exports a CSV,
summary JSON and `unrecognized_sidewall_candidates.ply`. The diagnostic selects
slender Unknown triangles directly adjacent to recognized cones and reports
snapshot face IDs, cone IDs, distances, normals and centroid coordinates.
It does not claim these faces are cones, and excludes distant/unseeded regions.

Validated by Release build and a CUDA cone-only run on examples/2.stl:
residual reserve 1250 proposals, 4 new cores; 89 cone patches and 8421 cone
faces versus 85/8238 before the reserve. Run wall time 13.3306 seconds.
The diagnostic export contains 364 candidate faces. Compiler reported two
size_t-to-int warnings in existing adjacency merge code.
