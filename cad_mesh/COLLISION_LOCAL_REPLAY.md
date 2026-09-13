# Collision comparison and bounded local replay

Collision witnesses now also identify the nearest original triangle to each
candidate triangle centroid, restricted to the same patch. The immutable baseline
is the mesh after shared boundary sampling and before any patch reconstruction.
`*_collision_source_pairs.ply` contains these original triangles, with pair IDs
matching `*_collision_pairs.ply`. Both files use the same world coordinates.

`nearest_source_pair_contact` is 1 if those two distinct source triangles intersect,
0 if they do not, and -1 if correspondence is unavailable or both map to one face.
This is local evidence, not a reliable inherited/new intersection classifier:
nearest-centroid correspondence does not identify the full source support. No
contact is automatically exempted based on this flag.

On the first collision attempt, affected Freeform patches are replayed from the
baseline, locking all original triangles overlapping witness boxes expanded by
max(target edge length, twice the deviation tolerance). All edges of protected
triangles are fixed. Other regions still undergo remeshing. This is conservative
spatial replay, not an exact operation-history undo. It performs one additional
remesh of affected patches and may protect more geometry than strictly necessary.
Successful replay is assembled and collision-checked again before acceptance.

If conflicts remain, one reconstructed side per witnessed pair is selected for
rollback, preferring fewer output faces and then lower patch ID. This is an explicit
recovery heuristic, not proof of which side introduced the collision. Rechecking
can reveal previously hidden pairs. After six attempts the existing full baseline
fallback remains as a bounded safety exit; ordinary attempts no longer roll back
both sides indiscriminately. Witnesses remain sampled, not an exhaustive contact set.

The earlier 174028 diagnostic directory was absent. The subsequently supplied
175204 directory was inspected; findings are in COLLISION_PAIR_FINDINGS_175204.md.
Both-triangles-exactly-unchanged source contacts now bypass rollback, with continued
search for other collisions. One-sided recovery prefers changed candidate geometry
over unchanged source geometry. Other source-contact flags remain diagnostic only.
No compilation, tests or remesh runs were performed. Exact causal operation undo
and general inherited-contact classification remain future work.
