param(
    [string]$InputStl = '',
    [string]$OutputDirectory = '',
    [ValidateRange(0.000001, 1000000000.0)][double]$TargetEdgeLength = 6.0,
    [ValidateRange(0.0, 1000000000.0)][double]$MaxDeviation = 0.1,
    [ValidateRange(0.000001, 180.0)][double]$MaxNormalDeviationDegrees = 10.0,
    [ValidateRange(1, 128)][int]$AnalyticWorkers = 4,
    [switch]$PatchDetails,
    [switch]$Cpu,
    [ValidateRange(1, 128)][int]$GenericRemeshWorkers = 20
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($InputStl)) {
    $InputStl = Join-Path $projectRoot 'examples/111.stl'
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $suffix = [guid]::NewGuid().ToString('N').Substring(0, 8)
    $OutputDirectory = Join-Path $projectRoot "examples/remesh_full_${stamp}_${suffix}"
}
$InputStl = [IO.Path]::GetFullPath($InputStl)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$executable = Join-Path $PSScriptRoot 'win/Release/cad_mesh_segment.exe'
foreach ($requiredFile in @($InputStl, $executable)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file is missing: $requiredFile"
    }
}

$env:CADMESH_CUDA_PROFILE = if ($Cpu) { '0' } else { '1' }
$env:CADMESH_CUDA_PROFILE_BY_TYPE = '0'
$env:CADMESH_REMESH_THREADS = "$AnalyticWorkers"
$env:CADMESH_REMESH_PATCH_DETAILS = if ($PatchDetails) { '1' } else { '0' }
$env:CADMESH_REMESH_CHART_CUDA = if ($Cpu) { '0' } else { '1' }
# Reset filters left by earlier surface-only runs in this PowerShell session.
$env:CADMESH_REMESH_SIMPLE_PLANES_ONLY = '0'
$env:CADMESH_REMESH_CYLINDERS_ONLY = '0'
$env:CADMESH_REMESH_CONES_ONLY = '0'
$env:CADMESH_REMESH_OTHER_FEATURES_ONLY = '0'
$culture = [Globalization.CultureInfo]::InvariantCulture
$nativeArguments = @(
    $InputStl, $OutputDirectory, '--remesh',
    '--generic-remesh-workers', "$GenericRemeshWorkers",
    '--target-edge-length', $TargetEdgeLength.ToString('R', $culture),
    '--max-deviation', $MaxDeviation.ToString('R', $culture),
    '--max-normal-deviation-degrees', $MaxNormalDeviationDegrees.ToString('R', $culture),
    '--collapse-passes', '0', '--flip-passes', '0', '--relax-iterations', '0'
)
if ($Cpu) {
    $nativeArguments += '--cpu'
} else {
    $nativeArguments += @('--analytic-seed-backend', 'cuda', '--require-remesh-cuda')
}
Write-Host "[remesh] Input: $InputStl"
Write-Host "[remesh] Output directory: $OutputDirectory"
Write-Host '[remesh] STL -> fresh partition -> shared boundaries -> patch remesh -> checks -> PLY. No partition snapshot/cache; no global postprocessing.'
$timer = [Diagnostics.Stopwatch]::StartNew()
try {
    & $executable @nativeArguments
    if ($LASTEXITCODE -ne 0) { throw "Native partition/remesh failed (exit $LASTEXITCODE)." }
} finally {
    $timer.Stop()
    Write-Host ("[remesh] End-to-end process time: {0:F3} s (STL import, partition, remesh and export)" -f $timer.Elapsed.TotalSeconds)
}
Write-Host "[remesh] Result: $(Join-Path $OutputDirectory 'remesh_result.ply')"
