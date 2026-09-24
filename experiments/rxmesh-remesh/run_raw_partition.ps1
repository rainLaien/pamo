param(
  [string]$InputMesh = "$PSScriptRoot/../../examples/Unnamed-Body.stl",
  [string]$SavedPartition = "",
  [string]$OutputDirectory = "$PSScriptRoot/results/raw-partition",
  [string]$Segmenter = "$PSScriptRoot/../../cad_mesh/win/Release/cad_mesh_segment.exe",
  [string]$Remesher = "$PSScriptRoot/build_rx/Release/cad_raw_partition_cli.exe",
  [ValidateRange(1,32)][int]$Workers = 4,
  [ValidateRange(1,8)][int]$GpuConcurrency = 4,
  [ValidateRange(0,32)][int]$PatchesPerTask = 0,
  [ValidateRange(0,256)][int]$ComputeRegions = 0,
  [ValidateRange(1,1000)][int]$Iterations = 20,
  [ValidateRange(1,12)][int]$SmoothPasses = 12,
  [ValidateRange(1,8)][int]$CollapsePasses = 8,
  [ValidateRange(1,8)][int]$FlipPasses = 8,
  [float]$TargetLength = 0,
  [float]$MaxError = 0,
  [int]$GpuMemoryMB = 0,
  [switch]$LegacyFlip,
  [switch]$AutoPartition,
  [switch]$FeatureRefine
)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory=(Resolve-Path -LiteralPath $OutputDirectory).Path
$totalTimer=[Diagnostics.Stopwatch]::StartNew()
$partitionSeconds=0.0
if($ComputeRegions -gt 0) {
  $partitionTimer=[Diagnostics.Stopwatch]::StartNew()
  $regionSource=if([string]::IsNullOrWhiteSpace($SavedPartition)){$InputMesh}else{$SavedPartition}
  $SavedPartition=Join-Path $OutputDirectory 'input.cadpart'
  if((Test-Path -LiteralPath $regionSource) -and
     (Resolve-Path -LiteralPath $regionSource).Path -eq $SavedPartition) {
    $SavedPartition=Join-Path $OutputDirectory 'compute.cadpart'
  }
  & python "$PSScriptRoot/tools/build_face_regions.py" $regionSource $SavedPartition --regions $ComputeRegions
  if($LASTEXITCODE -ne 0){throw 'Connected compute-region construction failed'}
  $partitionSeconds=$partitionTimer.Elapsed.TotalSeconds
} elseif([string]::IsNullOrWhiteSpace($SavedPartition)) {
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
$effectivePatchesPerTask=if($PatchesPerTask -gt 0){$PatchesPerTask}elseif($ComputeRegions -gt 0){1}else{0}
$arguments=@($SavedPartition,$output,'--workers',$Workers,'--gpu-concurrency',$GpuConcurrency,'--iters',$Iterations,'--smooth-passes',$SmoothPasses,'--collapse-passes',$CollapsePasses,'--flip-passes',$FlipPasses)
if($effectivePatchesPerTask -gt 0){$arguments+=@('--patches-per-task',$effectivePatchesPerTask)}
if($TargetLength -gt 0){$arguments+=@('--target',$TargetLength.ToString([Globalization.CultureInfo]::InvariantCulture))}
if($MaxError -gt 0){$arguments+=@('--max-error',$MaxError.ToString([Globalization.CultureInfo]::InvariantCulture))}
if($GpuMemoryMB -gt 0){$arguments+=@('--memory-mb',$GpuMemoryMB)}
if($LegacyFlip){$arguments+='--legacy-flip'}
if($AutoPartition){$arguments+='--auto-partition'}
if($FeatureRefine){$arguments+='--feature-refine'}
& $Remesher @arguments
$result=$LASTEXITCODE
@{partition_and_packaging_seconds=$partitionSeconds;total_seconds=$totalTimer.Elapsed.TotalSeconds;exit_code=$result;workers=$Workers;gpu_concurrency=$GpuConcurrency;compute_regions=$ComputeRegions;patches_per_task_override=$effectivePatchesPerTask;iterations=$Iterations;smooth_passes=$SmoothPasses;collapse_passes=$CollapsePasses;flip_passes=$FlipPasses;strict_flip_quality=(!$LegacyFlip);auto_partition_single_patch=([bool]$AutoPartition);snapshot=$SavedPartition} |
  ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $OutputDirectory 'pipeline.json')
if($result -eq 3){Write-Warning 'Some regions retained their input geometry. Inspect remeshed.ply.json; this is a partial result.';exit 3}
if($result -eq 4){Write-Warning 'Output quality regressed. Inspect remeshed.ply.json and remeshed.ply; this is not an accepted remesh.';exit 4}
if($result -ne 0){throw "Raw CUDA partition remesh failed: $result"}
Write-Output "Raw CUDA partition result: $output"
