param(
    [string]$PartitionDirectory = '',
    [string]$OutputDirectory = '',
    [ValidateRange(0.000001, 1000000000.0)][double]$TargetEdgeLength = 6.0,
    [ValidateRange(0.000001, 180.0)][double]$MaxNormalDeviationDegrees = 10.0,
    [ValidateRange(0.000001, 1.0)][double]$TargetMeanQuality = 0.8,
    [ValidateRange(0.0, 1000000000.0)][double]$MaxDeviation = 0.0,
    [ValidateRange(1, 10000)][int]$SplitPasses = 128,
    [ValidateRange(0, 10000)][int]$CollapsePasses = 12,
    [ValidateRange(0, 10000)][int]$FlipPasses = 32,
    [ValidateRange(0, 10000)][int]$RelaxIterations = 8
)

$ErrorActionPreference = 'Stop'
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
& $pythonExecutable (Join-Path $PSScriptRoot 'prepare_partition_snapshot.py') $PartitionDirectory $snapshot
if ($LASTEXITCODE -ne 0) { throw "Partition packaging failed (exit $LASTEXITCODE)." }

# Surface fitting is skipped entirely in this entry point.
$env:CADMESH_CUDA_PROFILE_BY_TYPE = '0'
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$nativeArguments = @(
    $snapshot, (Join-Path $OutputDirectory 'remesh_result.ply'), '--partition-snapshot', '--require-remesh-cuda',
    '--target-edge-length', $TargetEdgeLength.ToString('R', $culture),
    '--max-normal-deviation-degrees', $MaxNormalDeviationDegrees.ToString('R', $culture),
    '--target-mean-quality', $TargetMeanQuality.ToString('R', $culture),
    '--split-passes', "$SplitPasses", '--collapse-passes', "$CollapsePasses",
    '--flip-passes', "$FlipPasses", '--relax-iterations', "$RelaxIterations"
)
if ($PSBoundParameters.ContainsKey('MaxDeviation')) {
    $nativeArguments += @('--max-deviation', $MaxDeviation.ToString('R', $culture))
}
& $nativeExecutable @nativeArguments
if ($LASTEXITCODE -ne 0) { throw "Native remesh failed (exit $LASTEXITCODE)." }
Write-Host "[remesh] Result: $(Join-Path $OutputDirectory 'remesh_result.ply')"
