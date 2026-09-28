# Phase 37: locate short edges and connected low-quality regions

This is an independent diagnostic on the accepted phase-24 `2.stl` mesh,
using the byte-identical phase-34 sidecar. It does not change the algorithm,
target length, geometry budget, acceptance threshold, or output mesh.
The diagnostic reconstructs output edge incidence and low-quality face
components from the PLY and validates its totals against the batch report.

| Measure | Accepted output |
| --- | ---: |
| Native short edges | 676,458 |
| On protected/constrained output edges | 98,300 |
| Other output edges | 578,158 |
| Other short edges touching a pending patch | 70,028 |
| Faces with three short edges | 423,892 of 499,935 |
| Low-quality area | 4,727.793 |
| Within producer-labeled Fillet patches | 3,565.913 (75.4%) |
| Outside those Fillet patches | 1,161.880 |

The largest connected poor region (55.522 area, 203 faces) spans patches
3842 and 3843; 49.256 of that area is producer-labeled Fillet. The second
(53.096 area) is entirely within Fillet patch 3816. The first three patches
with the most unconstrained short-edge incidences (4277, 3815, 4878) are
large Freeform patches and are not marked pending. Their median output face
edge lengths are 0.611, 0.614, and 0.479 against a roughly 7.624 target;
their mean face quality is 0.633, 0.636, and 0.666. This is a genuine size
distribution mismatch by the current scalar target, although whether the
original surface permits coarsening remains unverified.

The producer's Fillet label is evidence of a relationship, not a CAD
certificate. The tallies above cannot justify moving Fillet/support stations
or exempting their low-quality faces from reporting. Conversely, a blanket
collapse pass aimed at all 676,458 short edges would spend most of its work
outside currently pending quality patches and could erase geometric detail.
The next candidate should use an original-STL geometry check and a small
selected region, with unchanged interface stations, before considering a
broader size-recovery operation. Fillet strips need a joint boundary-curve
contract before any station movement.

Reproduce:

```powershell
python experiments/rxmesh-remesh/results/phase37_short_edges.py experiments/rxmesh-remesh/results/phase34_provenance/stage/remeshed.ply experiments/rxmesh-remesh/results/phase34_provenance/stage/short_edges.json
```

Detailed component and patch counts are in
`results/phase34_provenance/stage/short_edges.json`. No commit or push was
made.
