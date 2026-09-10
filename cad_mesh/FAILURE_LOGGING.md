# Concise remesh logging

The global remesh path now disables per-chart success/progress/wait/backend
telemetry and boundary-split iteration logging. Local repartition success/island
reports and global Freeform success/wait reports are removed. CPU worker inner
iterations remain quiet. Basic stage and final status/time summaries remain.

AnalyticPatchRemeshReport stores failure diagnostics independently of verbose
printing. The global dispatcher emits only failed patches with the exact output
patch_id, source_patch_id, actual type, input face count, reason and detail.
Cylinder detail includes boundary loop count, construction stage and failure
string, plus offending global edge endpoints/length/target when fixed-boundary
sampling is required. Boundary tracing failures identify invalid_boundary_topology.
Other chart failures retain a generic construction category where deeper details
are not available. These are not silently labeled as precise causes.

No remesh acceptance rules changed. No build or runtime validation was performed.
