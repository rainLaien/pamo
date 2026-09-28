# Phase 39: investigate nearest-source normal anomalies

The accepted `2.stl` output has 456 face-centroid samples whose normal has
a nonpositive dot product with the nearest triangle from the same source
patch (total face area 95.597). This is a diagnostic warning, not by itself
a proof of 456 inverted or self-intersecting faces.

For each warning the diagnostic measured the distance to the nearest source
triangle in the same patch whose normal has a positive dot product with the
output face. It used the unchanged accepted PLY and repaired CADPART1.

| Observation | Count | Output face area |
| --- | ---: | ---: |
| All nonpositive nearest-normal samples | 456 | 95.597 |
| Positive-normal source match within 0.001 distance of the nearest face | 25 | 0.310 |
| Positive-normal source match within 0.05 distance | 45 | 57.947 |
| No positive-normal source triangle in the same patch | 215 | 18.316 |

The six largest anomalous output faces lie in patches 4995, 5031 and 5060;
each has area about 9.425 and nearest-normal dot about -0.141. Their
positive-normal source alternatives are only 0.009-0.033 farther than the
nearest source face, so source-face choice is material for those warnings.
The no-positive-match set is mostly tiny faces, but includes some faces up
to area 0.550. These observations neither certify the large faces nor permit
dismissing the remaining warnings. They identify where a topology-aware
surface correspondence or local geometric inspection is needed.

The new phase-38 ordinary second-pass warning has area 0.000482; its nearest
opposite-normal source face is 0.000710 away and the nearest aligned source
face is 0.034900 away. It is not an exact nearest-face tie, so rejecting
that candidate under the current orientation nonregression gate remains
prudent.

Diagnostic output: `results/phase39_orientation_ambiguity.json`. Reproduce:

```powershell
python experiments/rxmesh-remesh/results/phase39_orientation_ambiguity.py experiments/rxmesh-remesh/results/phase24_concurrency/full_gpu8/remeshed.ply experiments/rxmesh-remesh/results/phase24_concurrency/full_gpu8/strip_repair/pass_01/input.cadpart experiments/rxmesh-remesh/results/phase39_orientation_ambiguity.json
```

The run took 133.66 s as an offline audit. No algorithm, quality acceptance,
or output mesh was changed. The accepted full-flow time remains 103.09 s.
No commit or push was made.
