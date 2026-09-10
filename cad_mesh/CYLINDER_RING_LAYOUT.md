# Cylinder ring interior layout

The two-loop `direct_periodic_strip` path now resamples the interior independently
of the boundary vertices. The original trim vertices and their global aliases
remain fixed. Boundary breakpoint unions are used to integrate trim lengths and
check the ruled parameterization, not as mandatory columns in every interior row.

Interior columns use the accumulated maximum physical length of the two trims,
with spacing at most approximately 0.65 times the circumferential target. Rows
use the maximum physical gap between trims, at approximately 0.55 times that target.
Alternating rows have offset columns. Transition triangles connect these rows to
the original boundary samples; periodic seam copies retain shared vertex aliases.

The anisotropic metric remains in boundary compatibility and local edge repair.
Thus `axial_target` still describes the acceptance metric, while
`ring_interior_spacing` describes the new physical interior layout target.
Single-loop cylinder CDT is unchanged. This is not a quality-threshold iteration.
Small-radius tall cylinders may need more faces; existing mesh budgets still apply.

Diagnostics added:

- `ring_rows`, `ring_columns`, `ring_interior_spacing` (zero outside the ring path).
- `analytic commit timing`: future wait, detailed patch logging, lifting and
  orientation, and topology validation. Worker construction overlaps these times;
  do not add worker sums to the commit timers to estimate total wall time.
- `analytic collision timing`: accumulated index construction, queries, restoration.
- `post-analytic compact`: final array compaction.

No build, tests, or remesh run were performed for this change; validation is left
to the user as requested.
