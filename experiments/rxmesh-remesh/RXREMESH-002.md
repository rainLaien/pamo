# RXREMESH-002 — active-set dirty rings

Status: **COMPLETE (002.0–002.2; scope locked)**  
Date: 2026-09-14  
Depends on: RXREMESH-001 complete.

001 classifies **all edges** every iteration. 002 only classifies edges in the
dirty 2-ring of vertices that just split / collapsed / flipped / smoothed.

RXMesh does not learn CAD types. Dirty flags are topology only.

## Gates

### 002.0 CPU dirty 2-ring

- After an accepted operator, mark the vertex and its 2-ring dirty
- Next classify skips edges whose endpoints are both clean
- First iteration is fully dirty
- If an operator accepts nothing, dirty set is unchanged
- `test_cpu_isotropic` and `test_cpu_active_set` stay green

### 002.1 GPU dirty 2-ring

- Same rule on `EdgeAttribute` / vertex dirty mask
- Do not reset every edge to `Unseen` each iteration
- `test_rxmesh_isotropic` / `test_rxmesh_adaptive` stay green
- 1M plane: split+collapse+flip wall time drops vs 001 (record, do not gate)

### 002.2 Candidate counters on GPU

- Fill `split_candidates` / `collapse_candidates` / `flip_candidates` in JSON
- Reject reasons remain 0 until kernels log them

## NOT in 002

- PAMO patch graph (Phase 6)
- Cone / sphere / freeform
- Custom MIS / graph coloring (still RXMesh scheduler)

## Implementation notes

- CPU starts with every vertex dirty. After each operator batch with accepted
  changes, a deduplicated breadth-first walk replaces the active set with the
  touched vertices and their two topological rings on the rebuilt mesh. An empty
  accepted set preserves the previous mask, including across iterations.
- GPU uses `v_dirty` and `v_touched`. Both attributes travel through cavity
  migration and patch slicing. Two separate VV passes expand the touched mask;
  each pass reads an immutable source and writes a different destination.
  EV marks dirty-endpoint edges `Unseen` and clean-endpoint edges `Skip` before
  each operator. The RXMesh scheduler and its retry behavior are retained.
- Split/collapse initialize the new vertex dirty; flip touches all four diamond
  vertices. Smoothing copies clean vertices unchanged and touches only vertices
  whose coordinates change. Projection is restricted to touched vertices.
- GPU candidate counters count qualifying classification attempts before cavity
  scheduling (and before link-condition rejection for collapse/flip). A scheduler
  retry can count the same edge again; these are not unique-edge counts. The
  counters accumulate across iterations and are written to the existing JSON
  fields. Reject-reason counters stay zero.
- `smooth_moves` on GPU now counts vertex moves, rather than smoothing passes;
  a zero-displacement pass does not replace the dirty set.
- The two-ring helper finishes with a synchronized copy. Clearing touched is
  deferred to the next operator: launching an asynchronous reset immediately
  before a host read of managed counters triggers a Windows access fault.

## 1M plane observation (2026-09-14)

RTX 3060, driver 591.86, CUDA 12.6, Release (`sm_86;sm_89`). One run per
version, using `--grid 1000000 OUTPUT.obj 0.002 --gpu --iters 2` (actual input:
502,681 vertices / 1,002,528 triangles). The 001 executable was saved before
rebuilding the existing partial 002 sources. Both measurements overlapped
CPU-only CUDA linking; no other GPU benchmark ran concurrently.

| Metric | 001 baseline | 002 active set |
| --- | ---: | ---: |
| Split seconds | 0.186756 | 0.243425 |
| Collapse seconds | 2.151850 | 1.446450 |
| Flip seconds | 1.561420 | 1.126230 |
| **Split + collapse + flip seconds** | **3.900026** | **2.816105** |
| Total remesh seconds (including setup/export/metrics) | 16.4341 | 18.0643 |
| Accepted splits / collapses / flips | 1291 / 63205 / 87442 | 1286 / 36031 / 42141 |
| Mean quality | 0.896782 | 0.880057 |
| Mean sizing error | 0.194888 | 0.203090 |

The three topology stages decreased **27.8%** in this observation, while total
time increased. The finite-iteration outputs differ, so this is not an
equal-output or statistical speedup claim. Both outputs have valid topology,
held constraints, zero moved locked vertices and zero geometry error.
Performance remains non-gating.

Full reports, candidate counts, executable hashes and measurement caveats:
[`results/RXREMESH-002-plane-1m.json`](results/RXREMESH-002-plane-1m.json).
These timings identify the executable measured before the closed-STL flip
guard follow-up below.

## Validation

- CPU-only Ninja build: **7/7 tests passed**, including `test_cpu_isotropic`
  and `test_cpu_active_set`.
- CUDA Release build: **14/14 tests passed**, including
  `test_rxmesh_isotropic`, `test_rxmesh_adaptive`, candidate JSON checks and
  the new `test_rxmesh_active_set` (64.20 seconds for the complete suite).
- CPU regression coverage: exact two-ring extent, no-accept preservation,
  new-vertex inclusion, and repeated classification of rejected candidates.
- GPU regression coverage: exact two-ring masks across multiple RXMesh patches,
  clean-endpoint exclusion, fully dirty initialization, candidate counters,
  and no-op smoothing.

```powershell
cmake --build experiments/rxmesh-remesh/build --parallel 8
ctest --test-dir experiments/rxmesh-remesh/build --output-on-failure
cmake --build experiments/rxmesh-remesh/build_rx --config Release --parallel 8
ctest --test-dir experiments/rxmesh-remesh/build_rx -C Release --output-on-failure
```

## Closed-STL flip guard follow-up

The user's `examples/Unnamed-Body.stl`, run with the CLI defaults (`h=9.00539`,
five iterations), exposed a missing flip legality check. The preserved 001
executable also reproduced `export: non-manifold edge` on this input.

Operator isolation before the fix:

| Enabled operators | Export topology |
| --- | --- |
| Split only | Valid |
| Collapse only | Valid |
| Flip only | Non-manifold |
| Smooth only | Valid |
| Split + collapse | Valid |
| Split + collapse + flip | Non-manifold |
| Split + collapse + smooth | Valid |
| All four | Non-manifold |

The old common-neighbor check on the original edge endpoints did not exclude
an already existing edge between the opposite diamond vertices. A flip could
therefore add a duplicate diagonal. RXMesh's internal edge handles could still
be consistent, while the exported triangle list had more than two faces sharing
the same vertex pair.

The flip filter now explicitly rejects an existing opposite diagonal, scanning
both owned and ribbon edges before cavity creation. The closed input is retained
as `fixtures/closed_body.stl`; `test_rxmesh_closed_topology` covers flip-only and
the full pipeline by default, with additional operator combinations behind
`--stress`, and requires exactly two incident faces per exported edge.

Exploratory runs also exposed intermittent GPU memory errors during small-mesh
patch migration. Inputs with at most 512 faces now use an initial patch size of
512 instead of 256, keeping this input in one initial patch. General migration
robustness remains unverified.

Following the user's request, validation after this adjustment was limited to
one direct GPU CLI run on `examples/Unnamed-Body.stl`, with the original default
target length and five iterations. No stress tests were run. The rebuilt CLI
exited successfully and exported **367 vertices / 730 triangles** in 0.931456
reported seconds, with `topology_valid=true` and `constraints_held=true`.
Independent OBJ inspection found 1,095 edges, zero boundary edges, zero
non-manifold edges, zero duplicate faces, and Euler characteristic 2.
This is a successful result for the supplied file, not a general stability claim.

Output: `build_rx/Release/Unnamed-Body.gpu.fixed.obj`, with its adjacent JSON
report. Input/executable hashes and the export check are recorded in
[`results/RXREMESH-002-closed-body.json`](results/RXREMESH-002-closed-body.json).
