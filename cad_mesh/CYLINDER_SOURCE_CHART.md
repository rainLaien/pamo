# Trimmed cylinder source charts

Cylinder boundary construction now retries using the original patch triangles
when ear clipping or the monotone ring layout cannot construct a chart.
The retry remains a cylinder parameter-domain remesh, not a Freeform conversion.

The retry unwraps angular coordinates along source mesh edges and uses the
source triangles as the initial 2D triangulation. Boundary edges are recovered
from triangle incidence; their original global vertex IDs are fixed. Interior
diagonals can be legalized and split, with interior lattice points inserted
using the existing constrained-domain implementation. Holes are represented
by the actual boundary segments, without a one-loop/two-loop restriction.
UV height is scaled for the existing separate angular and axial target lengths
and restored before lifting back to the cylinder.

This removes boundary-only ear clipping as the sole option for irregularly
trimmed walls. Failure to trace simple boundary loops also permits a source
chart attempt. It does not relax boundary sampling or treat a folded UV mesh
as valid. A source mesh with nonzero angular cycle winding now enters the
periodic cut path in `PeriodicCylinderDomain.h`. It lifts each source triangle
into a consistent angular interval and keys vertex copies by source ID and
winding. Exposed source-internal edges form paired seam sides. Both sides use
the same subdivisions and negative aliases (starting at -2, since -1 denotes
an independent interior point in the native implementation).

Seam samples are inserted into incident source faces before native constrained
legalization, lattice insertion and edge repair. Physical trim boundaries keep
their original IDs; only artificial seams receive local paired samples.
The cut builder tries up to twelve meridians, and retains shared-vertex source
connectivity without requiring disjoint simple boundary loops. Source interior
edges are seed edges, not constraints. This differs from Python's boundary-only
Triangle input, but uses the same winding and paired-seam construction.
Resource exhaustion stops refinement retries; physical boundary sampling errors
are reported rather than independently changing a neighboring patch's edge.
Other failures distinguish
nonmanifold source edges, degenerate UV triangles and inconsistent orientation.
Failure logs retain the original boundary attempt's reason as well.

No build, tests or remesh runs were performed for this change, as requested.
