# Preserve good input regions before Freeform remeshing

This selection runs on the existing partition after shared boundary sampling. It
does not modify segmentation, require a new partition, or remove reference geometry.

Candidate triangles must have normalized triangle quality >= 0.6, longest edge
<= target edge length (plus numerical tolerance), and finite nonzero area. Only
edge-connected candidate islands with at least 8 faces are retained. This is one
input selection pass, not an output quality-threshold iteration. Candidates are
not certified for self-intersection by this selection.

The remaining edge-connected face components become independent repair regions.
Each region uses its own original triangles for projection and deviation checks.
Boundary IDs and original constraint edges stay fixed. Preserved triangles and
interfaces are assembled using original global IDs, including interfaces to a
region that fails. A failed region retains its input faces; successful regions
remain in a partial patch result. If every attempted region fails the patch falls
back. A patch made entirely of preserved islands needs no region operations.

Logs report kept faces, island count, preserved input area percentage, repair
region count and largest region size. Region failures and failed input area are
reported separately. `remeshed=2` is still a patch-level accepted processing status:
it can include deliberately preserved triangles and retained failed subregions.
Use the region summary to assess partial processing; per-face subregion status is
not yet exported. Final collision handling remains patch-level and may undo an
accepted partial patch. Exact original contact exemptions continue to apply.

Long skinny connected networks may remain one large repair region, so this is not
a guarantee of bounded region size. No arbitrary geometric cut is introduced.
Large regions retain detailed stage timing; small regions emit compact progress.

No compilation, tests or remesh runs were performed, as requested.
