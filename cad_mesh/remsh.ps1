param(
    [ValidateSet("cpu", "cuda")]
    [string]$ProjectionBackend = "cuda",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$AnalyticSeedBackend = "cuda",
    [ValidateRange(0.0, 180.0)][double]$MaxNormalDeviationDegrees = 10.0,
    [ValidateRange(1, 10000000)][int]$BatchFaceLimit = 75000,
    [switch]$TraceSurfaceValidation,
    [switch]$NoPartitionCache,
    [switch]$FullOutput,
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$moduleRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $moduleRoot
$pythonExecutable = Join-Path $projectRoot '.venv/Scripts/python.exe'
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $moduleRoot 'debug/112'
    if (Test-Path -LiteralPath $OutputDirectory) {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $suffix = [guid]::NewGuid().ToString('N').Substring(0, 8)
        $OutputDirectory = Join-Path $moduleRoot "debug/112_${stamp}_${suffix}"
    }
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
Write-Host "[remesh] Output directory: $OutputDirectory"

$remeshArguments = @(
    (Join-Path $moduleRoot 'remesh_file.py'), (Join-Path $projectRoot 'examples/111.stl'),
    '--output', $OutputDirectory,
    '--flip-passes', '8', '--target-edge-length', '12',
    '--projection-backend', $ProjectionBackend,
    '--analytic-seed-backend', $AnalyticSeedBackend,
    '--batch-face-limit', $BatchFaceLimit,
    '--max-normal-deviation-degrees',
    $MaxNormalDeviationDegrees.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture)
)
if ($TraceSurfaceValidation) {
    $remeshArguments += '--trace-surface-validation'
}
if ($NoPartitionCache) {
    $remeshArguments += '--no-partition-cache'
}
if ($FullOutput) {
    $remeshArguments += '--full-output'
}
& $pythonExecutable @remeshArguments
if ($LASTEXITCODE -ne 0) {
    throw "PaMO remesh failed (exit $LASTEXITCODE). See the error above."
}
