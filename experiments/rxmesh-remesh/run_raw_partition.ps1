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
  [ValidateRange(1,1000000)][int]$ModelSeeds = 2500,
  [ValidateRange(1,1000)][int]$Iterations = 20,
  [ValidateRange(1,12)][int]$SmoothPasses = 12,
  [ValidateRange(1,8)][int]$CollapsePasses = 8,
  [ValidateRange(1,8)][int]$FlipPasses = 8,
  [float]$TargetLength = 0,
  [float]$MaxError = 0,
  [ValidateRange(0,1)][float]$LowQualityThreshold = 0,
  [switch]$LegacyCoverageAcceptance,
  [switch]$GlobalQualityAcceptance,
  [switch]$ExploreProvisionalChildren,
  [switch]$TrackRegionCandidates,
  [switch]$SizeFeasibleFinalRefine,
  [switch]$SelectFinalRegions,
  [switch]$AuditFields,
  [string]$RegionalPatchTargets = '',
  [string]$RegionalReferenceOutput = '',
  [string]$RegionalQualityPolicy = '',
  [string]$NativeSizeEvidence = '',
  [switch]$GradedRegionalNeighbors,
  [switch]$UnifiedRegionalSizing,
  [int]$GpuMemoryMB = 0,
  [string]$Python = 'python',
  [ValidateRange(1,5)][int]$StripRepairPasses = 3,
  [switch]$SkipStripRepair,
  [switch]$LegacyFlip,
  [switch]$AutoPartition,
  [switch]$FeatureRefine,
  [ValidateRange(0.01,0.99)][float]$FeatureLengthRatio = 0.6,
  [ValidateRange(1,10)][float]$FeatureTransitionWidthRatio = 4,
  [Alias('CylinderCurvatureSizing')][switch]$CurvedSurfaceSizing
)
$ErrorActionPreference='Stop'
# PowerShell 7 can promote native stderr progress output to a terminating
# error. The segmenter reports normal import timing on stderr, so rely on
# the explicit $LASTEXITCODE checks below to detect native command failures.
if(Test-Path variable:PSNativeCommandUseErrorActionPreference){$PSNativeCommandUseErrorActionPreference=$false}
function Invoke-CheckedNative {
  param([string]$Executable,[object[]]$NativeArguments)
  $previousPreference=$ErrorActionPreference
  try {
    # Windows PowerShell 5 turns ordinary native stderr progress into a
    # PowerShell error. Exit codes below remain the source of failure status.
    $ErrorActionPreference='Continue'
    & $Executable @NativeArguments
    $script:LASTEXITCODE=$LASTEXITCODE
  } finally {
    $ErrorActionPreference=$previousPreference
  }
}
if($CurvedSurfaceSizing -and ($FeatureRefine -or $RegionalPatchTargets)){
  throw 'Curved surface sizing requires no feature refinement or explicit patch targets.'
}
if($GlobalQualityAcceptance -and $LegacyCoverageAcceptance){throw 'Global quality acceptance cannot be combined with legacy coverage acceptance.'}
if($RegionalQualityPolicy -and !$RegionalPatchTargets){throw 'Joint quality policy requires an explicit regional transaction.'}
if($NativeSizeEvidence -and !$RegionalQualityPolicy){throw 'Native size evidence requires a joint quality policy.'}
if($GradedRegionalNeighbors -and !$RegionalPatchTargets){throw 'Graded regional neighbors require an explicit regional selection.'}
if($UnifiedRegionalSizing -and (!$RegionalPatchTargets -or $FeatureRefine)){throw 'Unified regional sizing requires explicit regional targets without curvature refinement.'}
if($RegionalPatchTargets -and (!$RegionalReferenceOutput -or $SkipStripRepair -or $LegacyCoverageAcceptance)) {
  throw 'Regional candidates require a reference output, strip repair, and the quality acceptance policy.'
}
if($RegionalPatchTargets) {
  if(!(Test-Path -LiteralPath $RegionalPatchTargets) -or !(Test-Path -LiteralPath ($RegionalReferenceOutput+'.json'))) {throw 'Regional targets or reference report missing.'}
  if([IO.Path]::GetFullPath($RegionalReferenceOutput) -eq [IO.Path]::GetFullPath((Join-Path $OutputDirectory 'remeshed.ply'))) {throw 'Regional reference must remain immutable in a separate output directory.'}
  $referenceStats=Get-Content -LiteralPath ($RegionalReferenceOutput+'.json') -Raw | ConvertFrom-Json
  if($LowQualityThreshold -eq 0){$LowQualityThreshold=[float]$referenceStats.source_region_quality.threshold}
}
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
  Invoke-CheckedNative $Python @("$PSScriptRoot/tools/build_face_regions.py",$regionSource,$SavedPartition,'--regions',$ComputeRegions)
  if($LASTEXITCODE -ne 0){throw 'Connected compute-region construction failed'}
  $partitionSeconds=$partitionTimer.Elapsed.TotalSeconds
} elseif([string]::IsNullOrWhiteSpace($SavedPartition)) {
  $partitionTimer=[Diagnostics.Stopwatch]::StartNew()
  $directory=Join-Path $OutputDirectory 'partition'
  Invoke-CheckedNative $Segmenter @($InputMesh,$directory,'--remesh-handoff','--stop-after-partition','--max-model-seeds',$ModelSeeds)
  if($LASTEXITCODE -ne 0){throw 'PAMO partition failed'}
  $SavedPartition=Join-Path $OutputDirectory 'input.cadpart'
  Invoke-CheckedNative $Python @("$PSScriptRoot/../../cad_mesh/prepare_partition_snapshot.py",$directory,$SavedPartition)
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
if($FeatureRefine){$commonArguments+=@('--feature-refine','--feature-length-ratio',$FeatureLengthRatio.ToString('R',[Globalization.CultureInfo]::InvariantCulture),'--feature-transition-width-ratio',$FeatureTransitionWidthRatio.ToString('R',[Globalization.CultureInfo]::InvariantCulture))}
if($CurvedSurfaceSizing){$commonArguments+='--curved-surface-sizing'}
if($LegacyCoverageAcceptance){$commonArguments+='--legacy-coverage-acceptance'}
if($GlobalQualityAcceptance){$commonArguments+='--global-quality-acceptance'}
if($ExploreProvisionalChildren){$commonArguments+='--explore-provisional-children'}
if($TrackRegionCandidates){$commonArguments+='--track-region-candidates'}
if($SizeFeasibleFinalRefine){$commonArguments+='--size-feasible-final-refine'}
if($SelectFinalRegions){$commonArguments+='--select-final-regions'}
if($AuditFields){$commonArguments+='--audit-fields'}
$diagnosticThreshold=[double]$LowQualityThreshold
$regionalReferenceRetained=$false
$rounds=@()
$best=$null
$previousRepair=''
$residual=''
$curvedFallbackUsed=$false
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
    if($RegionalPatchTargets){$repairArguments+=@('--patch-targets',$RegionalPatchTargets)}
    if($GradedRegionalNeighbors){$repairArguments+='--graded-neighbors'}
    if($residual){$repairArguments+=@('--residual',$residual,'--previous-report',$previousRepair)}
    Invoke-CheckedNative $Python (@("$PSScriptRoot/tools/repair_narrow_strips.py")+$repairArguments)
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
  if($diagnosticThreshold -gt 0){$arguments+=@('--low-quality-threshold',$diagnosticThreshold.ToString('R',[Globalization.CultureInfo]::InvariantCulture))}
  if($UnifiedRegionalSizing) {
    if(!$repairInfo.regional_effective_patch_targets){throw 'Missing effective regional sizing targets.'}
    foreach($property in $repairInfo.regional_effective_patch_targets.PSObject.Properties) {
      $targetText=([double]$property.Value).ToString('R',[Globalization.CultureInfo]::InvariantCulture)
      $arguments+=@('--patch-target',($property.Name+':'+$targetText))
    }
    $arguments+=@('--sizing-gradation','0.5')
  }
  Invoke-CheckedNative $Remesher $arguments
  $roundResult=$LASTEXITCODE
  if(!(Test-Path -LiteralPath "$roundOutput.json")){throw "Raw CUDA partition remesh failed: $roundResult; no report at $roundOutput.json"}
  $stats=Get-Content -LiteralPath "$roundOutput.json" -Raw | ConvertFrom-Json
  $roundCurvedFallback=$false
  $roundEffectiveSizingMode=$(if($CurvedSurfaceSizing){'analytic_curvature'}elseif($FeatureRefine){'feature_aware'}else{'uniform'})
  if($CurvedSurfaceSizing -and [int]$stats.boundary_edges_deferred -gt 0) {
    $roundCurvedFallback=$true
    $curvedFallbackUsed=$true
    $baseArguments=@($arguments | Where-Object {$_ -ne '--curved-surface-sizing'})
    $packIndex=[Array]::IndexOf($baseArguments,'--patches-per-task')
    if($packIndex -ge 0){$baseArguments[$packIndex+1]=16}
    $featureOutput=Join-Path $roundDirectory 'remeshed_feature_fallback.ply'
    $featureArguments=@($baseArguments)
    $featureArguments[1]=$featureOutput
    $featureArguments+=@('--feature-refine','--feature-length-ratio',$FeatureLengthRatio.ToString('R',[Globalization.CultureInfo]::InvariantCulture),'--feature-transition-width-ratio',$FeatureTransitionWidthRatio.ToString('R',[Globalization.CultureInfo]::InvariantCulture))
    Write-Warning "曲面尺寸请求触及边界细分预算（$($stats.boundary_edges_deferred) 条边）；正在尝试特征驱动 remesh。"
    Invoke-CheckedNative $Remesher $featureArguments
    $roundResult=$LASTEXITCODE
    $featureStats=$null
    if(Test-Path -LiteralPath "$featureOutput.json"){$featureStats=Get-Content -LiteralPath "$featureOutput.json" -Raw | ConvertFrom-Json}
    if($roundResult -eq 0 -and $featureStats.quality_accepted -eq $true -and
       $featureStats.topology_valid -eq $true -and $featureStats.boundaries_held -eq $true -and
       [int]$featureStats.output_region_quality.long_edges -eq 0) {
      $roundOutput=$featureOutput
      $stats=$featureStats
      $roundEffectiveSizingMode='feature_aware'
    } else {
      $uniformOutput=Join-Path $roundDirectory 'remeshed_uniform_fallback.ply'
      $uniformArguments=@($baseArguments)
      $uniformArguments[1]=$uniformOutput
      Write-Warning '特征驱动候选未通过完整验收；正在用均匀尺寸重新 remesh。'
      Invoke-CheckedNative $Remesher $uniformArguments
      $roundResult=$LASTEXITCODE
      $roundOutput=$uniformOutput
      if(!(Test-Path -LiteralPath "$roundOutput.json")){throw "Uniform fallback remesh failed: $roundResult; no report at $roundOutput.json"}
      $stats=Get-Content -LiteralPath "$roundOutput.json" -Raw | ConvertFrom-Json
      $roundEffectiveSizingMode='uniform'
    }
  }
  if($roundResult -notin @(0,3,4)){throw "Raw CUDA partition remesh failed: $roundResult; inspect $roundOutput.json"}
  if($diagnosticThreshold -eq 0 -and $null -ne $stats.source_region_quality){$diagnosticThreshold=[double]$stats.source_region_quality.threshold}
  $rounds+=@{pass=$pass+1;repair_report=$repairReportPath;gpu_executed=$true;gpu_report="$roundOutput.json";exit_code=$roundResult;long_edges=$stats.long_edges_after_refine;max_edge_ratio=$stats.max_output_edge_ratio;curved_sizing_fallback=$roundCurvedFallback;effective_sizing_mode=$roundEffectiveSizingMode}
  $candidate=@{pass=$pass+1;output=$roundOutput;snapshot=$roundInput;stats=$stats;exit_code=$roundResult;target=$roundTarget;curved_fallback=$roundCurvedFallback;effective_sizing_mode=$roundEffectiveSizingMode}
  # Endpoint quality precedes size. A rejected candidate is retained for review
  # and residual repair, never promoted solely because it contains changed faces.
  $candidateQuality=($stats.quality_accepted -eq $true)
  $bestQuality=($null -ne $best -and $best.stats.quality_accepted -eq $true)
  $candidateBetter=($null -eq $best)
  if($null -ne $best) {
    if(!$LegacyCoverageAcceptance -and $candidateQuality -ne $bestQuality) {
      $candidateBetter=$candidateQuality
    } elseif(!$LegacyCoverageAcceptance -and $stats.output_region_quality.largest_low_quality_area -ne $best.stats.output_region_quality.largest_low_quality_area) {
      $candidateBetter=($stats.output_region_quality.largest_low_quality_area -lt $best.stats.output_region_quality.largest_low_quality_area)
    } elseif(!$LegacyCoverageAcceptance -and $stats.output_region_quality.low_quality_area -ne $best.stats.output_region_quality.low_quality_area) {
      $candidateBetter=($stats.output_region_quality.low_quality_area -lt $best.stats.output_region_quality.low_quality_area)
    } else {
      $candidateBetter=(($roundResult -eq 0 -and $best.exit_code -ne 0) -or
        ($roundResult -eq $best.exit_code -and
          ($stats.long_edges_after_refine -lt $best.stats.long_edges_after_refine -or
            ($stats.long_edges_after_refine -eq $best.stats.long_edges_after_refine -and $stats.max_output_edge_ratio -lt $best.stats.max_output_edge_ratio))))
    }
  }
  if($candidateBetter){$best=$candidate}
  if($SkipStripRepair){$stopReason='repair_disabled';break}
  # A global quality pass may still leave local defects. Give the existing
  # residual repair one more chance; its unchanged-candidate guard stops it.
  $pendingLocalWork=($GlobalQualityAcceptance -and @($stats.unresolved_patch_ids).Count -gt 0)
  if($stats.long_edges_after_refine -eq 0 -and ($LegacyCoverageAcceptance -or $candidateQuality) -and !$pendingLocalWork){$stopReason='endpoint_guards_satisfied';break}
  $residual=$roundOutput
  $previousRepair=$repairReportPath
}
if($null -eq $best){throw 'No remesh result was produced'}
Copy-Item -LiteralPath $best.output -Destination $output -Force
Copy-Item -LiteralPath ($best.output+'.json') -Destination ($output+'.json') -Force
$result=[int]$best.exit_code
if($RegionalPatchTargets) {
  $comparisonPath=Join-Path $OutputDirectory 'regional_comparison.json'
  $comparisonArguments=@('--source',$InputMesh,'--snapshot',$SavedPartition,'--repair-report',$rounds[$best.pass-1].repair_report,'--baseline',$RegionalReferenceOutput,'--candidate',$output,'--output',$comparisonPath,'--threshold',$diagnosticThreshold.ToString('R',[Globalization.CultureInfo]::InvariantCulture),'--target',$best.target.ToString('R',[Globalization.CultureInfo]::InvariantCulture),'--max-error',([double]$best.stats.max_geometry_error).ToString('R',[Globalization.CultureInfo]::InvariantCulture))
  if($RegionalQualityPolicy){$comparisonArguments+=@('--joint-quality-policy',$RegionalQualityPolicy)}
  if($NativeSizeEvidence){$comparisonArguments+=@('--native-size-evidence',$NativeSizeEvidence)}
  Invoke-CheckedNative $Python (@("$PSScriptRoot/tools/compare_regional_candidate.py")+$comparisonArguments)
  if($LASTEXITCODE -ne 0){throw 'Regional candidate comparison failed; candidate remains saved.'}
  $regionalComparison=Get-Content -LiteralPath $comparisonPath -Raw | ConvertFrom-Json
  if(!$regionalComparison.endpoint_accepted){$result=4}
  if(!$regionalComparison.regional_transaction_adoptable){
    # Roll back only this regional transaction. Keep the prior remesh, not the
    # original STL, and retain the rejected candidate for further processing.
    $provisional=Join-Path $OutputDirectory 'regional_provisional.ply'
    Copy-Item -LiteralPath $output -Destination $provisional -Force
    Copy-Item -LiteralPath ($output+'.json') -Destination ($provisional+'.json') -Force
    Copy-Item -LiteralPath $RegionalReferenceOutput -Destination $output -Force
    Copy-Item -LiteralPath ($RegionalReferenceOutput+'.json') -Destination ($output+'.json') -Force
    $regionalComparison.candidate=[IO.Path]::GetFullPath($provisional)
    $regionalComparison | Add-Member -NotePropertyName published_output -NotePropertyValue $output
    $regionalComparison | Add-Member -NotePropertyName output_action -NotePropertyValue 'reference_retained'
    $regionalComparison | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 -LiteralPath $comparisonPath
    $best.stats=Get-Content -LiteralPath ($output+'.json') -Raw | ConvertFrom-Json
    $regionalReferenceRetained=$true
    $result=4
  }
}
if($result -ne 4 -and !$SkipStripRepair -and $best.stats.long_edges_after_refine -gt 0){$result=3}
@{global_quality_acceptance=([bool]$GlobalQualityAcceptance);regional_quality_policy=$RegionalQualityPolicy;native_size_evidence=$NativeSizeEvidence;audit_fields_enabled=([bool]$AuditFields);audit_fields_candidate_prefix=$(if($AuditFields){$best.output}else{''});audit_fields_selected_prefix=$(if($AuditFields -and !$regionalReferenceRetained){$best.output}else{''});partition_and_packaging_seconds=$partitionSeconds;model_seeds=$ModelSeeds;total_seconds=$totalTimer.Elapsed.TotalSeconds;exit_code=$result;input_mesh=[IO.Path]::GetFullPath($InputMesh);target_length=$TargetLength;effective_target_length=$best.target;max_geometry_error=$MaxError;workers=$Workers;gpu_concurrency=$GpuConcurrency;compute_regions=$ComputeRegions;patches_per_task_override=$effectivePatchesPerTask;curved_surface_sizing_requested=([bool]$CurvedSurfaceSizing);curved_surface_sizing_effective=($best.effective_sizing_mode -eq 'analytic_curvature');feature_sizing_effective=($best.effective_sizing_mode -eq 'feature_aware');effective_sizing_mode=$best.effective_sizing_mode;curved_surface_sizing_fallback_used=$curvedFallbackUsed;iterations=$Iterations;smooth_passes=$SmoothPasses;collapse_passes=$CollapsePasses;flip_passes=$FlipPasses;strict_flip_quality=(!$LegacyFlip);auto_partition_single_patch=([bool]$AutoPartition);feature_refine=([bool]$FeatureRefine);snapshot=$SavedPartition;selected_snapshot=$(if($regionalReferenceRetained){''}else{$best.snapshot});candidate_snapshot=$best.snapshot;strip_repair_enabled=(!$SkipStripRepair);strip_repair_pass_limit=$passLimit;strip_repair_stop_reason=$stopReason;quality_accepted=($result -eq 0 -and $best.stats.quality_accepted -eq $true);graded_regional_neighbors=([bool]$GradedRegionalNeighbors);unified_regional_sizing=([bool]$UnifiedRegionalSizing);regional_patch_targets=$RegionalPatchTargets;regional_reference_output=$RegionalReferenceOutput;regional_reference_retained=$regionalReferenceRetained;regional_comparison_report=$(if($RegionalPatchTargets){$comparisonPath}else{''});low_quality_threshold=$diagnosticThreshold;legacy_coverage_acceptance=([bool]$LegacyCoverageAcceptance);explore_provisional_children=([bool]$ExploreProvisionalChildren);track_region_candidates=([bool]$TrackRegionCandidates);size_feasible_final_refine=([bool]$SizeFeasibleFinalRefine);select_final_regions=([bool]$SelectFinalRegions);selected_pass=$(if($regionalReferenceRetained){0}else{$best.pass});candidate_pass=$best.pass;remaining_long_edges=$best.stats.long_edges_after_refine;rounds=$rounds} |
  ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $OutputDirectory 'pipeline.json')
if($result -eq 3){Write-Warning 'Some regions or edge-size violations remain. Inspect pipeline.json and remeshed.ply.json; this is a partial result.';exit 3}
if($result -eq 4){Write-Warning 'Endpoint guards remain unsatisfied. Inspect pipeline.json and reports; any rejected regional candidate is retained separately.';exit 4}
if($result -ne 0){throw "Raw CUDA partition remesh failed: $result"}
Write-Output "Raw CUDA partition result: $output"
