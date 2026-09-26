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
  [string]$Python = 'python',
  [ValidateRange(1,5)][int]$StripRepairPasses = 3,
  [switch]$SkipStripRepair,
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
  & $Python "$PSScriptRoot/tools/build_face_regions.py" $regionSource $SavedPartition --regions $ComputeRegions
  if($LASTEXITCODE -ne 0){throw 'Connected compute-region construction failed'}
  $partitionSeconds=$partitionTimer.Elapsed.TotalSeconds
} elseif([string]::IsNullOrWhiteSpace($SavedPartition)) {
  $partitionTimer=[Diagnostics.Stopwatch]::StartNew()
  $directory=Join-Path $OutputDirectory 'partition'
  & $Segmenter $InputMesh $directory --remesh-handoff --stop-after-partition
  if($LASTEXITCODE -ne 0){throw 'PAMO partition failed'}
  $SavedPartition=Join-Path $OutputDirectory 'input.cadpart'
  & $Python "$PSScriptRoot/../../cad_mesh/prepare_partition_snapshot.py" $directory $SavedPartition
  if($LASTEXITCODE -ne 0){throw 'Partition packaging failed'}
  $partitionSeconds=$partitionTimer.Elapsed.TotalSeconds
}
$output=Join-Path $OutputDirectory 'remeshed.ply'
$SavedPartition=(Resolve-Path -LiteralPath $SavedPartition).Path
$effectivePatchesPerTask=if($PatchesPerTask -gt 0){$PatchesPerTask}elseif($ComputeRegions -gt 0){1}else{0}
$commonArguments=@('--workers',$Workers,'--gpu-concurrency',$GpuConcurrency,'--iters',$Iterations,'--smooth-passes',$SmoothPasses,'--collapse-passes',$CollapsePasses,'--flip-passes',$FlipPasses)
if($effectivePatchesPerTask -gt 0){$commonArguments+=@('--patches-per-task',$effectivePatchesPerTask)}
if($MaxError -gt 0){$commonArguments+=@('--max-error',$MaxError.ToString([Globalization.CultureInfo]::InvariantCulture))}
if($GpuMemoryMB -gt 0){$commonArguments+=@('--memory-mb',$GpuMemoryMB)}
if($LegacyFlip){$commonArguments+='--legacy-flip'}
if($AutoPartition){$commonArguments+='--auto-partition'}
if($FeatureRefine){$commonArguments+='--feature-refine'}
$rounds=@()
$best=$null
$previousRepair=''
$residual=''
$stopReason='pass_limit'
$passLimit=if($SkipStripRepair){1}else{$StripRepairPasses}
for($pass=0;$pass -lt $passLimit;$pass++) {
  $roundDirectory=Join-Path $OutputDirectory ('strip_repair/pass_{0:D2}' -f ($pass+1))
  New-Item -ItemType Directory -Force -Path $roundDirectory | Out-Null
  $roundInput=$SavedPartition
  $roundTarget=[double]$TargetLength
  $repairReportPath=''
  $repairInfo=$null
  if(!$SkipStripRepair) {
    $roundInput=Join-Path $roundDirectory 'input.cadpart'
    $repairReportPath=Join-Path $roundDirectory 'repair.json'
    $repairArguments=@($SavedPartition,$roundInput,'--report',$repairReportPath,'--target',$roundTarget.ToString('R',[Globalization.CultureInfo]::InvariantCulture))
    if($MaxError -gt 0){$repairArguments+=@('--max-error',$MaxError.ToString([Globalization.CultureInfo]::InvariantCulture))}
    if($residual){$repairArguments+=@('--residual',$residual,'--previous-report',$previousRepair)}
    & $Python "$PSScriptRoot/tools/repair_narrow_strips.py" @repairArguments
    if($LASTEXITCODE -ne 0){throw "Narrow-strip repair failed; inspect $repairReportPath"}
    $repairInfo=Get-Content -LiteralPath $repairReportPath -Raw | ConvertFrom-Json
    $roundTarget=[double]$repairInfo.target_length
    if($pass -gt 0 -and !$repairInfo.candidate_set_changed) {
      $stopReason='no_additional_supported_regions'
      $rounds+=@{pass=$pass+1;repair_report=$repairReportPath;gpu_executed=$false}
      break
    }
  }
  $roundOutput=Join-Path $roundDirectory 'remeshed.ply'
  $arguments=@($roundInput,$roundOutput)+$commonArguments
  if($roundTarget -gt 0){$arguments+=@('--target',$roundTarget.ToString('R',[Globalization.CultureInfo]::InvariantCulture))}
  & $Remesher @arguments
  $roundResult=$LASTEXITCODE
  if($roundResult -ne 0 -and $roundResult -ne 3){throw "Raw CUDA partition remesh failed: $roundResult; inspect $roundOutput.json"}
  $stats=Get-Content -LiteralPath "$roundOutput.json" -Raw | ConvertFrom-Json
  $rounds+=@{pass=$pass+1;repair_report=$repairReportPath;gpu_executed=$true;gpu_report="$roundOutput.json";exit_code=$roundResult;long_edges=$stats.long_edges_after_refine;max_edge_ratio=$stats.max_output_edge_ratio}
  $candidate=@{pass=$pass+1;output=$roundOutput;snapshot=$roundInput;stats=$stats;exit_code=$roundResult;target=$roundTarget}
  # Prefer complete runs, then fewer oversized edges, then a shorter worst edge.
  if($null -eq $best -or
     ($roundResult -eq 0 -and $best.exit_code -ne 0) -or
     ($roundResult -eq $best.exit_code -and
      ($stats.long_edges_after_refine -lt $best.stats.long_edges_after_refine -or
       ($stats.long_edges_after_refine -eq $best.stats.long_edges_after_refine -and $stats.max_output_edge_ratio -lt $best.stats.max_output_edge_ratio)))) {
    $best=$candidate
  }
  if($SkipStripRepair){$stopReason='repair_disabled';break}
  if($stats.long_edges_after_refine -eq 0){$stopReason='edge_limits_satisfied';break}
  $residual=$roundOutput
  $previousRepair=$repairReportPath
}
if($null -eq $best){throw 'No remesh result was produced'}
Copy-Item -LiteralPath $best.output -Destination $output -Force
Copy-Item -LiteralPath ($best.output+'.json') -Destination ($output+'.json') -Force
$result=[int]$best.exit_code
if(!$SkipStripRepair -and $best.stats.long_edges_after_refine -gt 0){$result=3}
@{partition_and_packaging_seconds=$partitionSeconds;total_seconds=$totalTimer.Elapsed.TotalSeconds;exit_code=$result;input_mesh=[IO.Path]::GetFullPath($InputMesh);target_length=$TargetLength;effective_target_length=$best.target;max_geometry_error=$MaxError;workers=$Workers;gpu_concurrency=$GpuConcurrency;compute_regions=$ComputeRegions;patches_per_task_override=$effectivePatchesPerTask;iterations=$Iterations;smooth_passes=$SmoothPasses;collapse_passes=$CollapsePasses;flip_passes=$FlipPasses;strict_flip_quality=(!$LegacyFlip);auto_partition_single_patch=([bool]$AutoPartition);feature_refine=([bool]$FeatureRefine);snapshot=$SavedPartition;selected_snapshot=$best.snapshot;strip_repair_enabled=(!$SkipStripRepair);strip_repair_pass_limit=$passLimit;strip_repair_stop_reason=$stopReason;selected_pass=$best.pass;remaining_long_edges=$best.stats.long_edges_after_refine;rounds=$rounds} |
  ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $OutputDirectory 'pipeline.json')
if($result -eq 3){Write-Warning 'Some regions or edge-size violations remain. Inspect pipeline.json and remeshed.ply.json; this is a partial result.';exit 3}
if($result -eq 4){Write-Warning 'Output quality regressed. Inspect remeshed.ply.json and remeshed.ply; this is not an accepted remesh.';exit 4}
if($result -ne 0){throw "Raw CUDA partition remesh failed: $result"}
Write-Output "Raw CUDA partition result: $output"
