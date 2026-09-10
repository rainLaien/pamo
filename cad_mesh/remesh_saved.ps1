param(
    [string]$PartitionDirectory = '',
    [string]$OutputDirectory = '',
    [ValidateRange(1, 16)][int]$AnalyticWorkers = 4,
    [switch]$CpuAnalytic,
    [Alias('SimplePlanesOnly')][switch]$PlanesOnly,
    [switch]$CylindersOnly,
    [switch]$ConesOnly,
    [switch]$OtherFeaturesOnly,
    [switch]$PatchDetails,
    [ValidateRange(0.000001, 1000000000.0)][double]$TargetEdgeLength = 6.0,
    [ValidateRange(0.000001, 180.0)][double]$MaxNormalDeviationDegrees = 10.0,
    [ValidateRange(0.0, 180.0)][double]$GenericFeatureAngleDegrees = 10.0,
    [ValidateRange(1, 100)][int]$GenericRemeshIterations = 5,
    [ValidateRange(1, 128)][int]$GenericRemeshWorkers = 20,
    [ValidateRange(0.000001, 1.0)][double]$TargetMeanQuality = 0.8,
    [ValidateRange(0.0, 1000000000.0)][double]$MaxDeviation = 0.1,
    [ValidateRange(1, 10000)][int]$SplitPasses = 128,
    # Compatibility arguments; patch remesh does not run global postprocessing.
    [ValidateRange(0, 10000)][int]$CollapsePasses = 0,
    [ValidateRange(0, 10000)][int]$FlipPasses = 0,
    [ValidateRange(0, 10000)][int]$RelaxIterations = 0
)

$ErrorActionPreference = 'Stop'
if (([int]$PlanesOnly.IsPresent + [int]$CylindersOnly.IsPresent + [int]$ConesOnly.IsPresent + [int]$OtherFeaturesOnly.IsPresent) -gt 1) {
    throw 'Choose only one of -PlanesOnly, -CylindersOnly, -ConesOnly, or -OtherFeaturesOnly.'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PartitionDirectory)) {
    $PartitionDirectory = Join-Path $projectRoot 'examples/partition_review'
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $suffix = [guid]::NewGuid().ToString('N').Substring(0, 8)
    $OutputDirectory = Join-Path $projectRoot "examples/remesh_saved_${stamp}_${suffix}"
}
$PartitionDirectory = [System.IO.Path]::GetFullPath($PartitionDirectory)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$pythonExecutable = Join-Path $projectRoot '.venv/Scripts/python.exe'
$nativeExecutable = Join-Path $PSScriptRoot 'win/Release/cad_mesh_segment.exe'
foreach ($requiredFile in @($pythonExecutable, $nativeExecutable,
        (Join-Path $PartitionDirectory 'patch_result.ply'),
        (Join-Path $PartitionDirectory 'patch_report.json'))) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file is missing: $requiredFile"
    }
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$snapshot = Join-Path $OutputDirectory 'partition_input.bin'
Write-Host "[remesh] Saved partition: $PartitionDirectory"
Write-Host "[remesh] Output directory: $OutputDirectory"
Write-Host '[remesh] Pipeline: shared boundary sampling -> preserve/repartition Others -> global patch remesh -> compact and export.'
if (-not ($PlanesOnly -or $CylindersOnly -or $ConesOnly -or $OtherFeaturesOnly)) {
    Write-Host '[remesh] Selection: all surface types; original Plane/Cylinder reconstruction enabled.'
}
if ($CollapsePasses -ne 0 -or $FlipPasses -ne 0 -or $RelaxIterations -ne 0) {
    Write-Host '[remesh] CollapsePasses, FlipPasses and RelaxIterations are ignored by the patch-only pipeline.'
}
& $pythonExecutable (Join-Path $PSScriptRoot 'prepare_partition_snapshot.py') $PartitionDirectory $snapshot
if ($LASTEXITCODE -ne 0) { throw "Partition packaging failed (exit $LASTEXITCODE)." }

# Initial whole-model fitting is skipped; selected Others are locally repartitioned.
$env:CADMESH_CUDA_PROFILE_BY_TYPE = '0'
$env:CADMESH_REMESH_THREADS = "$AnalyticWorkers"
$env:CADMESH_REMESH_PATCH_DETAILS = if ($PatchDetails) { '1' } else { '0' }
$env:CADMESH_REMESH_CHART_CUDA = if ($CpuAnalytic) { '0' } else { '1' }
$env:CADMESH_REMESH_SIMPLE_PLANES_ONLY = if ($PlanesOnly) { '1' } else { '0' }
$env:CADMESH_REMESH_CYLINDERS_ONLY = if ($CylindersOnly) { '1' } else { '0' }
$env:CADMESH_REMESH_CONES_ONLY = if ($ConesOnly) { '1' } else { '0' }
$env:CADMESH_REMESH_OTHER_FEATURES_ONLY = if ($OtherFeaturesOnly) { '1' } else { '0' }
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$nativeArguments = @(
    $snapshot, (Join-Path $OutputDirectory 'remesh_result.ply'), '--partition-snapshot', '--require-remesh-cuda',
    '--target-edge-length', $TargetEdgeLength.ToString('R', $culture),
    '--max-normal-deviation-degrees', $MaxNormalDeviationDegrees.ToString('R', $culture),
    '--generic-feature-angle-deg', $GenericFeatureAngleDegrees.ToString('R', $culture),
    '--generic-remesh-iterations', $GenericRemeshIterations.ToString($culture),
    '--generic-remesh-workers', $GenericRemeshWorkers.ToString($culture),
    '--target-mean-quality', $TargetMeanQuality.ToString('R', $culture),
    '--split-passes', "$SplitPasses", '--collapse-passes', "$CollapsePasses",
    '--flip-passes', "$FlipPasses", '--relax-iterations', "$RelaxIterations"
)
$nativeArguments += @('--max-deviation', $MaxDeviation.ToString('R', $culture))
& $nativeExecutable @nativeArguments
if ($LASTEXITCODE -ne 0) { throw "Native remesh failed (exit $LASTEXITCODE)." }
Write-Host "[remesh] Result: $(Join-Path $OutputDirectory 'remesh_result.ply')"
