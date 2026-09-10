$env:CADMESH_CUDA_PROFILE = '1'
$env:CADMESH_CUDA_PROFILE_BY_TYPE = '0'

& ./cad_mesh/win/Release/cad_mesh_segment.exe `
  ./examples/111.stl ./examples/partition_review `
  --remesh-handoff `
  --stop-after-partition `
  --target-edge-length 6 `
  --max-normal-deviation-degrees 10 `
  --target-mean-quality 0.8 `
  --require-remesh-cuda `
  --collapse-passes 12 --flip-passes 32 --relax-iterations 8
