param(
  [string]$InputMesh = "$PSScriptRoot/../../examples/Unnamed-Body.stl",
  [string]$SavedPartition = "",
  [string]$OutputDirectory = "$PSScriptRoot/results/raw-partition",
  [string]$Segmenter = "$PSScriptRoot/../../cad_mesh/win/Release/cad_mesh_segment.exe",
  [string]$Remesher = "$PSScriptRoot/build_rx/Release/cad_raw_partition_cli.exe",
  [ValidateRange(1,32)][int]$Workers = 2,
  [ValidateRange(1,1000)][int]$Iterations = 20,
  [float]$TargetLength = 0,
  [float]$MaxError = 0,
  [int]$GpuMemoryMB = 0,
  [switch]$FeatureRefine
)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory=(Resolve-Path -LiteralPath $OutputDirectory).Path
$totalTimer=[Diagnostics.Stopwatch]::StartNew()
$partitionSeconds=0.0
if([string]::IsNullOrWhiteSpace($SavedPartition)) {
  $partitionTimer=[Diagnostics.Stopwatch]::StartNew()
  $directory=Join-Path $OutputDirectory 'partition'
  & $Segmenter $InputMesh $directory --remesh-handoff --stop-after-partition
  if($LASTEXITCODE -ne 0){throw 'PAMO partition failed'}
  $SavedPartition=Join-Path $OutputDirectory 'input.cadpart'
  & python "$PSScriptRoot/../../cad_mesh/prepare_partition_snapshot.py" $directory $SavedPartition
  if($LASTEXITCODE -ne 0){throw 'Partition packaging failed'}
  $partitionSeconds=$partitionTimer.Elapsed.TotalSeconds
}
$output=Join-Path $OutputDirectory 'remeshed.ply'
$arguments=@($SavedPartition,$output,'--workers',$Workers,'--iters',$Iterations)
if($TargetLength -gt 0){$arguments+=@('--target',$TargetLength.ToString([Globalization.CultureInfo]::InvariantCulture))}
if($MaxError -gt 0){$arguments+=@('--max-error',$MaxError.ToString([Globalization.CultureInfo]::InvariantCulture))}
if($GpuMemoryMB -gt 0){$arguments+=@('--memory-mb',$GpuMemoryMB)}
if($FeatureRefine){$arguments+='--feature-refine'}
& $Remesher @arguments
$result=$LASTEXITCODE
@{partition_and_packaging_seconds=$partitionSeconds;total_seconds=$totalTimer.Elapsed.TotalSeconds;exit_code=$result;workers=$Workers;snapshot=$SavedPartition} |
  ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $OutputDirectory 'pipeline.json')
if($result -eq 3){Write-Warning 'Some regions retained their input geometry. Inspect remeshed.ply.json; this is a partial result.';exit 3}
if($result -ne 0){throw "Raw CUDA partition remesh failed: $result"}
Write-Output "Raw CUDA partition result: $output"
