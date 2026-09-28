<#
.SYNOPSIS
从原始 STL 运行分区、预处理、CUDA remesh、输出及原生尺寸复核。
.EXAMPLE
.\run_remesh.ps1
.EXAMPLE
.\run_remesh.ps1 -InputMesh 'D:\models\part.stl' -TargetLength 8.9 -MaxError 0.5
.EXAMPLE
.\run_remesh.ps1 -InputMesh .\examples\3.stl -CurvedSurfaceSizing -TargetLength 8.9206295 -MaxError 1.7841259
.EXAMPLE
.\run_remesh.ps1 -InputMesh .\examples\2.stl -FeatureAwareSizing -TargetLength 7.6237063 -MaxError 1.5247413
.NOTES
默认 examples/3.stl 与当前仓库使用的浇道.stl 相同。
尺寸和误差使用模型坐标单位。0 表示沿用入口自动设置。
只要网格和汇总成功保存，脚本退出码为 0。尺寸、分区和质量问题记录在 summary.json、pipeline.json 与 run.log。
无法生成网格的执行错误仍返回 1。
不自动编译、不复用分区、不覆盖已有结果。详细几何独立审查未包含在本脚本中。
曲面尺寸选项针对圆柱和圆锥使用单片区计算，以避免已观测到的多片区候选局部折叠；
该选项仍需针对新输入独立检查真实特征、方向和原始 STL 几何误差。
特征尺寸选项根据采样曲率及受保护特征线设置有界局部尺寸。
#>
[CmdletBinding()]
param(
  [string]$InputMesh = "$PSScriptRoot/../../examples/3.stl",
  [string]$OutputDirectory = '',
  [ValidateRange(0,1e20)][double]$TargetLength = 0,
  [ValidateRange(0,1e20)][double]$MaxError = 0,
  [ValidateRange(1,32)][int]$Workers = 8,
  [ValidateRange(1,8)][int]$GpuConcurrency = 8,
  [ValidateRange(1,1000)][int]$Iterations = 12,
  [ValidateRange(1,1000000)][int]$ModelSeeds = 2500,
  [Alias('CylinderCurvatureSizing')][switch]$CurvedSurfaceSizing,
  [switch]$FeatureAwareSizing,
  [ValidateRange(1,10)][double]$FeatureTransitionWidthRatio = 4,
  [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
# PowerShell 7 may turn the intentionally preserved exit code 4 into an exception.
if(Test-Path variable:PSNativeCommandUseErrorActionPreference){$PSNativeCommandUseErrorActionPreference=$false}
function Invoke-CheckedNative {
  param([string]$Executable,[object[]]$NativeArguments)
  $previousPreference=$ErrorActionPreference
  try {
    # Windows PowerShell 5 treats ordinary native stderr as an error record.
    # Native exit codes are checked by the caller after each invocation.
    $ErrorActionPreference='Continue'
    & $Executable @NativeArguments
    $script:LASTEXITCODE=$LASTEXITCODE
  } finally {
    $ErrorActionPreference=$previousPreference
  }
}
try {
  if($FeatureAwareSizing -and $CurvedSurfaceSizing){throw '请选择一种尺寸模式：FeatureAwareSizing 或 CurvedSurfaceSizing。'}
  if(!(Test-Path -LiteralPath $InputMesh -PathType Leaf)){throw "找不到输入文件：$InputMesh"}
  $InputMesh=(Resolve-Path -LiteralPath $InputMesh).Path
  if([IO.Path]::GetExtension($InputMesh) -ine '.stl'){throw '本脚本要求原始 STL 输入。'}
  $runner=Join-Path $PSScriptRoot 'run_raw_partition.ps1'
  $remesher=Join-Path $PSScriptRoot 'build_rx/Release/cad_raw_partition_cli.exe'
  $segmenter=Join-Path $PSScriptRoot '../../cad_mesh/win/Release/cad_mesh_segment.exe'
  foreach($file in @($runner,$remesher,$segmenter)){
    if(!(Test-Path -LiteralPath $file -PathType Leaf)){throw "缺少程序或入口：$file"}
  }
  Get-Command $Python -ErrorAction Stop | Out-Null
  if(!$OutputDirectory){$OutputDirectory=Join-Path $PSScriptRoot ('results/run_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))}
  $OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
  if(Test-Path -LiteralPath $OutputDirectory){
    if(!(Test-Path -LiteralPath $OutputDirectory -PathType Container) -or
       @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count){throw "输出目录已存在且非空，请换一个目录：$OutputDirectory"}
  }
  New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
  $log=Join-Path $OutputDirectory 'run.log'
  $shellExe=(Get-Process -Id $PID).Path
  $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$runner,
    '-InputMesh',$InputMesh,'-OutputDirectory',$OutputDirectory,
    '-Workers',$Workers,'-GpuConcurrency',$GpuConcurrency,'-PatchesPerTask',$(if($CurvedSurfaceSizing){1}else{16}),
    '-ModelSeeds',$ModelSeeds,'-Iterations',$Iterations,'-SmoothPasses',3,
    '-CollapsePasses',8,'-FlipPasses',8,'-SelectFinalRegions','-AuditFields','-GlobalQualityAcceptance','-Python',$Python)
  if($TargetLength -gt 0){$arguments+=@('-TargetLength',$TargetLength.ToString('R',[Globalization.CultureInfo]::InvariantCulture))}
  if($MaxError -gt 0){$arguments+=@('-MaxError',$MaxError.ToString('R',[Globalization.CultureInfo]::InvariantCulture))}
  if($CurvedSurfaceSizing){$arguments+='-CurvedSurfaceSizing'}
  if($FeatureAwareSizing){$arguments+='-FeatureRefine'}
  $arguments+=@('-FeatureTransitionWidthRatio',$FeatureTransitionWidthRatio.ToString('R',[Globalization.CultureInfo]::InvariantCulture))
  Write-Host "输入：$InputMesh"
  Write-Host "从原始 STL 开始计算，详细日志：$log"
  $timer=[Diagnostics.Stopwatch]::StartNew()
  Invoke-CheckedNative $shellExe $arguments *> $log
  $childExit=$LASTEXITCODE
  $pipelinePath=Join-Path $OutputDirectory 'pipeline.json'
  $mesh=Join-Path $OutputDirectory 'remeshed.ply'
  if(!(Test-Path -LiteralPath $pipelinePath) -or !(Test-Path -LiteralPath $mesh)){
    throw "计算未生成完整输出（进程码 $childExit），请查看 $log"
  }
  $pipeline=Get-Content -LiteralPath $pipelinePath -Raw | ConvertFrom-Json
  $report=Get-Content -LiteralPath ($mesh+'.json') -Raw | ConvertFrom-Json
  $acceptanceCode=[int]$pipeline.exit_code
  if($acceptanceCode -notin @(0,3,4)){throw "计算失败，状态 $acceptanceCode；请查看 $log"}
  $sizeAudit=Join-Path $OutputDirectory 'native_size_audit.json'
  $sizeAuditData=$null
  $sizeAuditStatus='unavailable'
  $auditExit=$null
  try {
    $previousPreference=$ErrorActionPreference
    try {
      $ErrorActionPreference='Continue'
      & $Python (Join-Path $PSScriptRoot 'tools/audit_native_fields.py') --mesh $pipeline.audit_fields_selected_prefix --snapshot $pipeline.selected_snapshot --output $sizeAudit >> $log 2>&1
      $auditExit=$LASTEXITCODE
    } finally {$ErrorActionPreference=$previousPreference}
    if($auditExit -eq 0 -and (Test-Path -LiteralPath $sizeAudit -PathType Leaf)) {
      $sizeAuditData=Get-Content -LiteralPath $sizeAudit -Raw | ConvertFrom-Json
      if($sizeAuditData.verified -eq $true){$sizeAuditStatus='verified'}
    }
  } catch {Add-Content -LiteralPath $log -Value "[run-remesh] Native size audit exception: $_"}
  if($sizeAuditStatus -ne 'verified'){
    Add-Content -LiteralPath $log -Value "[run-remesh] Native size audit unavailable; process code: $auditExit"
    Write-Warning "原生尺寸复核未完成；网格已保存，请查看 $log"
  }
  $coverageAudit=Join-Path $OutputDirectory 'coverage_audit.json'
  $coverageAuditData=$null
  $coverageAuditStatus='unavailable'
  $auditExit=$null
  try {
    $previousPreference=$ErrorActionPreference
    try {
      $ErrorActionPreference='Continue'
      & $Python (Join-Path $PSScriptRoot 'tools/audit_remesh_coverage.py') --mesh $mesh --snapshot $pipeline.selected_snapshot --output $coverageAudit >> $log 2>&1
      $auditExit=$LASTEXITCODE
    } finally {$ErrorActionPreference=$previousPreference}
    if($auditExit -eq 0 -and (Test-Path -LiteralPath $coverageAudit -PathType Leaf)) {
      $coverageAuditData=Get-Content -LiteralPath $coverageAudit -Raw | ConvertFrom-Json
      $coverageAuditStatus='completed'
    }
  } catch {Add-Content -LiteralPath $log -Value "[run-remesh] Coverage audit exception: $_"}
  if($coverageAuditStatus -ne 'completed'){
    Add-Content -LiteralPath $log -Value "[run-remesh] Coverage audit unavailable; process code: $auditExit"
    Write-Warning "重划覆盖率复核未完成；网格已保存，请查看 $log"
  }
  $reviewVisuals=Join-Path $OutputDirectory 'review_visuals.json'
  if($coverageAuditStatus -eq 'completed') {
    try {
      $previousPreference=$ErrorActionPreference
      try {
        $ErrorActionPreference='Continue'
        & $Python (Join-Path $PSScriptRoot 'tools/render_remesh_review.py') --mesh $pipeline.audit_fields_selected_prefix --snapshot $pipeline.selected_snapshot --coverage $coverageAudit --output-dir $OutputDirectory >> $log 2>&1
        $renderExit=$LASTEXITCODE
      } finally {$ErrorActionPreference=$previousPreference}
    } catch {
      $renderExit=1
      Add-Content -LiteralPath $log -Value "[run-remesh] Review rendering exception: $_"
    }
    if($renderExit -ne 0){Write-Warning "网格可视化未生成（进程码 $renderExit）；数值审查仍保留在 $coverageAudit"}
  }
  $q=$report.output_region_quality
  $summary=[ordered]@{
    input=$InputMesh;output=$mesh;input_sha256=(Get-FileHash -LiteralPath $InputMesh -Algorithm SHA256).Hash
    full_pipeline_seconds=$pipeline.total_seconds;pipeline_plus_size_audit_seconds=$timer.Elapsed.TotalSeconds
    partition_and_packaging_seconds=$pipeline.partition_and_packaging_seconds
    target_length=$report.effective_target_length;max_geometry_error=$report.max_geometry_error
    model_seeds=$ModelSeeds;curved_surface_sizing_requested=([bool]$CurvedSurfaceSizing)
    feature_sizing_requested=([bool]$FeatureAwareSizing)
    feature_transition_width_ratio=$FeatureTransitionWidthRatio
    curved_surface_sizing_effective=([bool]$pipeline.curved_surface_sizing_effective)
    feature_sizing_effective=([bool]$pipeline.feature_sizing_effective)
    effective_sizing_mode=$pipeline.effective_sizing_mode
    curved_surface_sizing_fallback_used=([bool]$pipeline.curved_surface_sizing_fallback_used)
    patches_per_task=$(if($pipeline.curved_surface_sizing_effective){1}else{16})
    output_faces=$report.output_faces;quality_mean=$q.mean;quality_p05=$q.p05
    low_quality_cutoff=$q.threshold;low_quality_area=$q.low_quality_area
    largest_low_quality_region_area=$q.largest_low_quality_area;low_quality_components=$q.low_quality_components
    native_long_edges=$q.long_edges;native_short_edges=$q.short_edges;native_short_constrained_edges=$q.short_constrained_edges
    native_target_min=$sizeAuditData.target_min;native_target_p05=$sizeAuditData.target_p05
    adjacent_target_growth_p99=$sizeAuditData.adjacent_target_growth_p99
    adjacent_target_growth_max=$sizeAuditData.adjacent_target_growth_max
    adjacent_target_growth_over_1_3=$sizeAuditData.adjacent_target_growth_over_1_3
    transition_faces=$sizeAuditData.transition_faces
    transition_skinny_faces=$sizeAuditData.transition_skinny_faces
    transition_skinny_area=$sizeAuditData.transition_skinny_area
    transition_quality_p05=$sizeAuditData.transition_quality_p05
    circular_plane_remeshed=$report.circular_plane_remeshed
    circular_plane_skipped=$report.circular_plane_skipped
    circular_planar_patch_count=$coverageAuditData.likely_circular_planar_patch_count
    circular_planar_without_output_interior_count=$coverageAuditData.circular_planar_without_output_interior_count
    circular_planar_mean_output_interior_vertices=$coverageAuditData.circular_planar_mean_output_interior_vertices
    retained_planar_patch_count=$coverageAuditData.retained_planar_patch_count
    retained_planar_source_faces=$coverageAuditData.retained_planar_source_faces
    retained_planar_area=$coverageAuditData.retained_planar_area
    retained_planar_area_fraction=$coverageAuditData.retained_planar_area_fraction
    coverage_complete=($coverageAuditStatus -eq 'completed' -and $coverageAuditData.retained_planar_patch_count -eq 0 -and
      $coverageAuditData.circular_planar_without_output_interior_count -eq 0)
    coverage_audit=$(if($coverageAuditStatus -eq 'completed'){$coverageAudit}else{''})
    coverage_audit_status=$coverageAuditStatus;native_size_audit_status=$sizeAuditStatus
    review_visuals=$(if(Test-Path -LiteralPath $reviewVisuals){$reviewVisuals}else{''})
    fallback_gpu_tasks=$report.fallback;accepted_gpu_tasks=$report.accepted
    topology_valid=$report.topology_valid;quality_accepted=$pipeline.quality_accepted
    pending_patch_count=@($report.pending_patch_ids).Count;unresolved_patch_count=@($report.unresolved_patch_ids).Count
    independent_original_geometry_audit_run=$false;native_size_audit=$(if($sizeAuditStatus -eq 'verified'){$sizeAudit}else{''})
    exit_code=0;acceptance_code=$acceptanceCode;result_status=$(if($acceptanceCode -eq 0 -and $sizeAuditStatus -eq 'verified' -and $coverageAuditStatus -eq 'completed'){'accepted'}else{'saved_with_warnings'})
    pipeline_report=$pipelinePath;detail_report=($mesh+'.json');log=$log
  }
  $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
  Write-Host ('完整 remesh 耗时：{0:N2} 秒' -f $pipeline.total_seconds)
  Write-Host ('三角形：{0}；平均质量：{1:N4}；P05：{2:N4}' -f $report.output_faces,$q.mean,$q.p05)
  Write-Host ('最长边超限：{0}；过短边：{1}；仍不达标分区：{2}' -f $q.long_edges,$q.short_edges,@($report.unresolved_patch_ids).Count)
  Write-Host "网格：$mesh"
  Write-Host "汇总：$(Join-Path $OutputDirectory 'summary.json')"
  if($pipeline.curved_surface_sizing_fallback_used){Write-Warning "曲面尺寸触及安全预算；输出采用 $($pipeline.effective_sizing_mode) 模式，请查看 summary.json。"}
  if($coverageAuditStatus -eq 'completed' -and !$summary.coverage_complete){Write-Warning "仍有 $($summary.retained_planar_patch_count) 个平面片区保持原有三角形；覆盖率明细见 coverage_audit.json。"}
  if($acceptanceCode -ne 0){
    $acceptanceMessage="[run-remesh] Mesh saved with acceptance code $acceptanceCode; long edges=$($q.long_edges); unresolved patches=$(@($report.unresolved_patch_ids).Count)"
    Add-Content -LiteralPath $log -Value $acceptanceMessage
    Write-Warning "网格已保存；尺寸或分区仍有未达标项，详情见 summary.json 和 run.log。"
  }
  exit 0
} catch {
  Write-Error $_ -ErrorAction Continue
  exit 1
}
