# CPU execution

After rebuilding the native executable, run from the project root:

```powershell
.\cad_mesh\cpu.ps1 -InputStl .\examples\222_li.stl -AnalyticWorkers 8 -GenericRemeshWorkers 20
```

This uses the same fresh partition, shared boundary and patch remesh pipeline.
The script defaults to examples/3.stl and target edge length 6, matching cuda.ps1.
Analytic reconstruction uses the existing patch worker pool (1..128 workers).
Freeform remeshing uses the existing independent region pool (1..128 workers).
Initial segmentation and secondary fitting use the existing CPU backend; this
change does not parallelize competitive region growth or model ownership.

The executable's `--cpu` switch forces CPU analytic fitting after option parsing,
disables CUDA boundary classification and prevents chart CUDA calls even if the
chart environment switch is enabled. Secondary fitting explicitly overrides any
saved segmentation backend. `--require-remesh-cuda` conflicts with `--cpu`.
The switch also works with `--partition-snapshot` when invoking the executable
directly. Existing saved-partition wrappers still require CUDA by default.

No build, tests or remesh execution performed for this change.
