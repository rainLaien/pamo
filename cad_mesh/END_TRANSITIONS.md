# Ruled-surface end transition candidates

The cylinder-only and cone-only diagnostic workflows now extract end
transition candidates after strict sidewall classification and before storing
Unknown residuals. Full multi-primitive partition retains its existing workflow.
Default ModelRuledGrowthDistance is 0.02 input units, superseding the previous
0.8 classification default; the CLI override remains available. This change
does not set any remesh deviation parameter.

Each mother surface supplies its observed axial bounds and local radius.
Unowned adjacent faces near either end seed a bounded search requiring distance
outside the sidewall tolerance, smooth signed neighboring normals, and a
progressive departure from the mother. A connected candidate needs at least
four faces, two traversal layers and five degrees of normal change. End
proximity, travel, turning and evaluation limits prevent unrestricted growth.
Competing mothers leave ambiguous faces unresolved. All original faces remain.
Boundaries follow existing mesh edges; no triangle is cut.

Candidates become independent patches. Up to 24 candidate groups attempt a
sampled torus fit followed by full-group certification; others remain Freeform
with reference-mesh projection. Feature role 2 (`FilletCandidate`) explicitly
distinguishes these heuristic extractions from certified fillets (role 1).
PLY, JSON, native snapshot reading and Python packaging support the new role.
Existing support-boundary analysis can upgrade candidates to Fillet when its
independent criteria pass.

This version does not guarantee a complete fillet to its tangent plane. Very
coarse facets, incomplete mothers, ambiguous junctions or budget limits can
leave transitions unresolved. Flat end caps should remain outside the turning
band. A full circumferential loop is not required.

Release build and CUDA run on examples/2.stl completed successfully. Output
examples/cones_fillets_20260911_163930 contains 75 cone patches (9801 faces),
191 Freeform/FilletCandidate patches (84108 faces) and 2068 Unknown patches.
All 781127 faces remain accounted for and the output partition validates.
Total wall time was 14.4105 seconds, transition extraction 0.16868 seconds.
No torus candidate passed the bounded fits on this model. This is not a
geometric accuracy validation of the candidate boundaries; many small Unknown
fragments remain around the extracted regions.
