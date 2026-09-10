# Flat remesh patches and one global Freeform pool

Saved input partition labels are now source metadata only for the remesh stage.
All output regions, including untouched source patches and preserved islands, are
registered in a flat OutputPatches table with new unique zero-based IDs.

Preparation completes before any remesh tasks are scheduled:

1. Selection is evaluated using the ORIGINAL source type. In Others mode original
   Plane/Cylinder patches remain unselected.
2. Selected Others triangles satisfying target edge length (with the existing
   numerical slack) are removed from the repair set. Each connected preserved
   island gets an independent output patch and status 3.
3. Remaining faces are independently model-fitted using the initial model-first
   engine and original resolution. Each returned region gets a new output patch.
   Preparation still respects original source interfaces; no cross-source fitting
   or merging is introduced.
4. Plane/Cylinder and other analytic children are rebuilt together using their
   actual new types. Original type-based filters are NOT reapplied to children:
   a plane recognized inside Others must be processed in Others mode.
5. ALL Freeform children enter one worker pool across source patch boundaries,
   default 20 workers. Larger input tasks are submitted first. The bounded window
   limits running/queued/completed-but-unmerged jobs to the worker count. Assembly
   follows deterministic task order. An immutable position snapshot prevents
   concurrent reallocation races while the caller appends generated vertices.

PLY face properties:

- patch_id: new flat output patch ID; IDs may differ from earlier exports.
- source_patch_id: original saved partition patch ID.
- primitive_type: actual child model type (preserved islands retain source type).
- remeshed: 0 unselected, 1 construction failure, 2 rebuilt, 3 preserved by length.
- remesh_reason: per-output-region reason, including 9 for preserved input.

Shared edges use original global vertex IDs and are fixed after preparation.
New child interfaces are not independently resampled. Generic sizing still uses
the VCGLib 0.8 / 4/3 length interval, not an absolute maximum output edge length.
An indivisible large Freeform patch remains one job. Model fitting preparation
remains sequential; parent grouping no longer limits remesh scheduling.

The output registry's TriangleIds describe final output faces. NeighborPatchIds
describe registered shared interfaces; stale source boundary/support IDs are
cleared. Status counts and reason areas use the new registry. Output contains the
registry directly, so writePly remains compatible with callers passing old source
metadata: it prefers result.OutputPatches when present.

Run: .\cad_mesh\remesh_others.ps1 -GenericRemeshWorkers 20
Logs: global patch registration / global freeform pool / global freeform complete.
No compilation, tests or remesh execution were performed.
