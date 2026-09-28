# Phase 34: retain producer patch roles and support relationships

The production remesh geometry and movement permissions are unchanged. This
is a limited audit-data change that preserves CADPART1 patch type, producer
Fillet role, and support patch IDs when `--audit-fields` is requested. It
does not assert CAD certification or classify an unlabeled interface as a
safe computation-only seam.

## Finding that changes the boundary plan

The accepted `2.stl` repaired source snapshot contains 6810 patches. Its
producer marks 1508 as Fillet and records 3016 support relationships. Of
92885 exported source constraint edges, 12213 lie on a declared
Fillet/support interface, but only 491 of those carry the input hard bit.
Thus 11722 producer-declared Fillet/support interface segments have hard=0.
The hard bit alone is not a valid compute-seam classifier.

At the difficult region, patch 3842 is a Freeform Fillet with support patches
607 and 3843. In this source snapshot, 607/3842 has 50 interface segments,
all hard=0; 3842/3843 has 25, all hard=0. The second pair was previously
called a *plausible* computational seam because its local source dihedral is
near zero. The producer support relationship supplies contrary evidence:
it may be a tangent Fillet boundary and must remain protected until a
shared-curve motion contract is established. Interface 489/3842 has 30
hard=0 segments and a roughly 14° source crease but is not listed as a
3842 support relationship; its origin still cannot be certified from this
data. The source 489/3843 interface has four hard=1 segments. None of these
facts gives permission to move or cross an interface.

## Implementation and verification

- `ConstraintAudit.h`, `PartitionInput.cpp`, and `ConstraintAudit.cpp` retain
  and export the immutable patch evidence to `*.source_patches.tsv` and a
  version-2 manifest. Export validates patch IDs, role range, and support
  references. A failed export still leaves `complete:false`.
- `audit_native_fields.py` independently compares every exported role and
  support list against the CADPART1 bytes, then counts Fillet/support
  interface segments and their hard bits. It continues to accept saved
  version-1 sidecars for historical comparisons.
- The ordinary loader and remesher do not use the new data to change an
  operator, target size, acceptance gate, or boundary permission.

The Release CLI and constraint-audit test were rebuilt. The C++
`constraint_audit` case and five Python native-field cases passed. A
same-parameter batch run on the phase-24 repaired `2.stl` snapshot used
workers 8, GPU concurrency 8, target 7.6237063, original geometry budget
1.5247413, iterations 12, smoothing 3, collapse/flip passes 8, regional
final selection, and global quality acceptance. Its final PLY SHA256 is
`f5d0b7222e451ce5d7676bf9919bea53fb5d5ba933eee4946147cbae4a9bf996`,
**byte-identical** to the accepted phase-24 mesh. The independent version-2
sidecar audit passed and reproduced native short/long counts 676458/0.

This was a reused-partition stage run. Its batch time was 48.38 s with
audit fields versus 43.49 s in the earlier phase-24 run without them;
single-run variation and optional export work prevent attributing the
difference to this source change. It is not a new raw-STL complete-flow
timing or a speed improvement. The latest accepted complete-flow evidence
remains 103.086 s, above the 90 s target.

The next geometry work can resample a declared Fillet support curve only
with a joint two-sided transaction that retains curve identity, corner and
hole connectivity, original-STL error budget, and native size transition.
It cannot gain permission by changing `hard=0` into “compute seam.”

## Reproduce

```powershell
cmake --build experiments/rxmesh-remesh/build_rx --config Release --target cad_raw_partition_cli test_constraint_audit --parallel 8
ctest --test-dir experiments/rxmesh-remesh/build_rx -C Release -R '^constraint_audit$' --output-on-failure
python -m unittest discover -s experiments/rxmesh-remesh/tests -p test_native_fields_audit.py
experiments/rxmesh-remesh/build_rx/Release/cad_raw_partition_cli.exe experiments/rxmesh-remesh/results/phase24_concurrency/full_gpu8/strip_repair/pass_01/input.cadpart experiments/rxmesh-remesh/results/phase34_provenance/stage/remeshed.ply --workers 8 --gpu-concurrency 8 --iters 12 --smooth-passes 3 --collapse-passes 8 --flip-passes 8 --max-error 1.5247413 --target 7.6237063 --low-quality-threshold 0.026690566912293434 --select-final-regions --global-quality-acceptance --audit-fields
python experiments/rxmesh-remesh/tools/audit_native_fields.py --mesh experiments/rxmesh-remesh/results/phase34_provenance/stage/remeshed.ply --snapshot experiments/rxmesh-remesh/results/phase24_concurrency/full_gpu8/strip_repair/pass_01/input.cadpart --output experiments/rxmesh-remesh/results/phase34_provenance/stage/native_fields_audit.json
```

The stage artifacts are under `results/phase34_provenance/`. No commit or
push was made.
