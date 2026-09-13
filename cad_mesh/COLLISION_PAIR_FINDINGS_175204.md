# Collision pair inspection: run 175204

Inspected existing candidate and source diagnostic PLY files in
`examples/remesh_full_20260910_175204_397_e3b98117`. No remesh, build or tests were run.
The baseline is the partition mesh after shared boundary sampling, before patch
reconstruction. It is not necessarily identical to raw STL topology.

There are 114 sampled pairs: 57 from attempt 1, 53 from attempt 2 and 4 from attempt 3.
Nearest-centroid source pair flags are 60 contact, 51 no contact, 3 unresolved.
These flags alone cannot establish inherited versus new intersection.

## Confirmed inherited geometry

Pairs 1 and 58 are patch 448 versus 2804. Both candidate triangles match their
corresponding source triangles (baseline face IDs 718687 and 718685), vertex for
vertex irrespective of ordering. Intersecting their plane-slice segments gives
length approximately 5.03627e-6, centered at
(2852.04312578, -5.39871138, -41.84537792). Baseline and candidate give the same
segment. Restoring a Freeform region cannot remove this pre-existing intersection.

## Changed geometry requiring further treatment

Patch 2315 participates in 21 distinct partner-patch samples in attempt 1.
Pair 0 (1946 versus 2315) has a finite intersection segment of about 1.19842.
The 1946 candidate is unchanged from its source triangle. In attempt 2, pairs
97, 98, 99, 100 and 108 still intersect patch 2315 while the opposing triangles
match their source geometry. Their intersection lengths are about 0.139, 0.469,
0.350, 0.405 and 0.396 respectively. This points to changed 2315 geometry as the
first side to restore, not automatic blame on the opposing Freeform patch.
Nearest source pairs are nonintersecting in these samples, but complete support
correspondence would be required to prove all these contacts newly introduced.

Attempt 3 also contains same-patch finite intersections in 2700 and 2791, with
segment lengths about 0.225 and 0.0589. These remain unresolved; proximity to
the reference surface does not itself rule out self-intersection.

## Code adjustment

Collision detection now exempts only pairs for which BOTH triangles are exactly
unchanged from same-patch baseline geometry. It continues searching after an
exempt hit so that it cannot mask another changed intersection. Exact matching
has no tolerance-based geometric relaxation and may conservatively miss a match
when the nearest-centroid source triangle is not the matching one.

For remaining witnessed conflicts, prefer a reconstructed side whose triangle
changed over an exactly unchanged side. If both changed, the previous bounded
recovery heuristic remains; this is not exact operation-causality tracing.
