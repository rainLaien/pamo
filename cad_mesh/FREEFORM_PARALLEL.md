# Freeform CPU subregion workers

After a parent's preservation selection and model-first repartition, independent
Freeform children are submitted to a native worker pool. Each task owns its points,
triangles, adjacency, reference index and output. Source positions, source faces,
models and constraints are read-only until the pool finishes. Analytic children
continue on the existing chart path on the calling thread.

Default GenericRemeshWorkers is 20 (allowed 1..128). Actual workers are capped by
the number of Freeform children. A sliding submission window limits the sum of
running, queued and completed-but-unassembled jobs to the worker count. Results
are assembled in original child order by the caller, retaining shared vertex IDs
and per-face status/reason. Worker exceptions propagate via futures; pool teardown
joins outstanding jobs before captured mesh data can be destroyed.

This is a thread pool, not 20 external processes. It parallelizes children within
one parent. Parent processing and its model-first fitting remain sequential,
avoiding nested fitting/GPU context concurrency. A single huge child is still a
single CPU task. Ordered assembly can briefly wait for a slow early child; this
bounded window intentionally limits result-memory buildup.

```powershell
.\cad_mesh\remesh_others.ps1 -GenericRemeshWorkers 20
```

Executable flag: --generic-remesh-workers. Logs: freeform worker pool,
freeform commit waiting, freeform worker result. Worker-level inner iteration
logging is disabled to avoid interleaving output; slow jobs report their duration
when assembled. in_flight includes completed results still awaiting assembly.

No compilation, tests or remesh execution were performed. Speedup and memory use
have not been measured; more workers do not guarantee a faster run.
