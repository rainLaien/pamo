# Cylinder-only diagnostic partition

Use `--cylinders-only` with the input STL and output directory. This flag
automatically enables partition-only snapshot export and rejects legacy mode,
remesh and saved-partition input. The default full partition is unchanged.

`examples/partition_cylinders_only.ps1` wraps the existing partition script;
it accepts InputStl, OutputDirectory and AnalyticSeedBackend (cpu/cuda/auto).
It does not build the executable.

The pipeline builds topology and sampling cells, proposes cylinders from ruled
strips and neighborhoods, certifies cylinder cores, and performs bounded ring
absorption with fixed models. Sampling cells do not assign planar patches.
No cone, sphere or torus candidates, generic residual fitting, seam bridging
or remesh are run. Failed-neighborhood suppression is disabled in this mode
so one failure does not suppress neighboring seeds, within the existing seed
budget. Core fitting thresholds remain unchanged; absorption permits 30-degree
normal error with strict vertex distance checks. Hard adjacency barriers and
the existing proposal/ring budgets still apply.

All remaining faces are stored in connected Unknown patches: primitive_type=0.
Recognized cylinders have primitive_type=2. Unknown does not mean proven
Freeform. The PLY and JSON snapshot preserve all partition faces. The cylinder
patch count is a count of recognized cores, not necessarily distinct physical
cylinders. This diagnostic does not guarantee complete cylinder recognition.

No compilation or runtime validation was performed for this change.
