# Reference-surface acceptance guard

Generic collapse, flip and vertex-movement candidates now check three vertices,
three edge midpoints and the centroid against the immutable input region.
Moved interior vertices are projected before checking incident triangles.
Candidate normals are checked against the nearest source triangle at the
centroid using MaximumNormalDeviationDegrees. Rejected operations do not commit.
Midpoint edge subdivision preserves the current piecewise-planar surface.

Before assembling output, all rebuilt analytic and generic child regions pass
the same sampled distance/normal checks against their own input child region.
The reference uses the boundary-sampled input before patch remeshing; it is not
the fitted analytic surface, nor an unrestricted nearest point on another wall.
An invalid child retains its original faces and gets remeshed=1, reason=4.
Unaffected children remain rebuilt. This final safety fallback is per child,
not arbitrary replacement of isolated faces (which would break conformity).

This restores acceptance protection without enabling the old pre-construction
model-deviation veto or whole-model collision rollback. Analytic automatic
refinement after rejection is not implemented by this change. Sampled checks
are not a continuous or bidirectional Hausdorff guarantee, and do not establish
absence of self-intersection. Shape error uses MaximumDeviation; classification
distance and target edge length are separate parameters.
