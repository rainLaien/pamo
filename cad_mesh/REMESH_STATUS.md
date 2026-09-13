# Current PLY status values

`remeshed` is now a three-state face attribute inherited from the patch:

| Value | Meaning |
| --- | --- |
| 0 | Patch not selected for this run |
| 1 | Patch selected, but not successfully processed (including skipped unsupported/nonanalytic patches, failed construction and collision rollback) |
| 2 | Processing accepted after final collision handling |

The internal RemeshedPatches mask remains boolean; only exported status is three-state.
`remesh_reason` retains its existing values (0 means success there). Older documents
describing `remeshed=1` as success refer to the former two-state export format.

Use `cad_mesh/remesh_others.ps1` for the current investigation. It reads the existing
partition from `examples/partition_review` by default and selects all types except
Plane and Cylinder. It does not fit or segment STL. Shared boundary sampling still
updates both sides of an interface; unselected patch interiors remain unchanged.
Saved snapshot packaging and loading are included in the wrapper timer.

No compilation, tests or remesh execution performed for this change.
