param(
  [string]$InputMesh = "$PSScriptRoot/../../examples/Unnamed-Body.stl",
  [string]$OutputDirectory = "$PSScriptRoot/results/Unnamed-Body-cad",
  [string]$Segmenter = "$PSScriptRoot/../../cad_mesh/win/Release/cad_mesh_segment.exe",
  [string]$Remesher = "",
  [float]$TargetLength = 0,
  [float]$MaxError = 0,
  [ValidateRange(1, 1000)][int]$Iterations = 20,
  [ValidateSet('global', 'gpu', 'cpu')][string]$Backend = 'global',
  [ValidateRange(0.001, 89.999)][float]$NormalDegrees = 10
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Remesher)) {
  $executableName = if ($Backend -eq 'global') { 'cad_adaptive_global_cli.exe' } else { 'cad_adaptive_cli.exe' }
  $Remesher = Join-Path $PSScriptRoot "build_rx/Release/$executableName"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$partitionDirectory = Join-Path $OutputDirectory 'partition'
$snapshotPath = Join-Path $OutputDirectory 'input.cadpart'
$outputPath = Join-Path $OutputDirectory 'remeshed.ply'
& $Segmenter $InputMesh $partitionDirectory --remesh-handoff --stop-after-partition
if ($LASTEXITCODE -ne 0) { throw 'PAMO geometric partition failed' }
& python "$PSScriptRoot/../../cad_mesh/prepare_partition_snapshot.py" $partitionDirectory $snapshotPath
if ($LASTEXITCODE -ne 0) { throw 'Partition packaging failed' }
$backendOption = '--' + $Backend
# Exit 3 means a geometrically validated candidate was saved, but convergence
# targets were not reached. It must not be reported as a successful remesh.
if (-not (Test-Path -LiteralPath $Remesher -PathType Leaf)) { throw "Remesher not found: $Remesher" }
$remeshExit = -1
try {
  $ErrorActionPreference = 'Continue'
  & $Remesher $snapshotPath $outputPath $TargetLength --partition $backendOption --iters $Iterations --max-error $MaxError --normal-degrees $NormalDegrees
  $remeshExit = $LASTEXITCODE
} finally {
  $ErrorActionPreference = 'Stop'
}
if ($remeshExit -ne 0 -and -not ($Backend -eq 'global' -and $remeshExit -eq 3)) {
  throw "Remesh backend '$Backend' failed with exit code $remeshExit"
}
& python "$PSScriptRoot/tools/check_cad_output.py" $partitionDirectory $outputPath --save (Join-Path $OutputDirectory 'validation.json')
if ($LASTEXITCODE -ne 0) { throw 'Exported feature/topology validation failed' }
if ($remeshExit -eq 3) {
  Write-Warning "CAD candidate saved but not converged: $outputPath. See remeshed.json; do not treat this as final quality acceptance."
  exit 3
}
Write-Output "CAD remesh output: $outputPath"
