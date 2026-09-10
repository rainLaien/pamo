# Default patch remesh pipeline

Run `cad_mesh/remesh_saved.ps1` without a surface filter to select all existing
analytic reconstruction paths. Surface-only switches still restrict selection.

The native pipeline samples shared boundaries, reconstructs patches, performs
patch topology checks and collision rollback, then compacts and exports. It no
longer invokes global interior splitting, collapse, flip, relaxation or global
quality measurement after reconstruction. The reference index used solely for
those removed postprocessing calls is no longer built.

Failed, unsupported and unselected patches retain their original interiors, with
shared boundary splits retained. No claim is made that those interiors meet the
target edge length. PLY `remeshed=1` identifies accepted patch reconstruction
after collision rollback. Per-type logs report selection, eligibility, acceptance,
fallback and unsupported/nonanalytic counts.

`SplitPasses` controls shared boundary sampling. Collapse/flip/relax arguments
remain accepted for command compatibility but do not enable global postprocessing.
No build, tests or remesh validation were run for this change, as requested.
