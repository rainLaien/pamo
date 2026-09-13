param(
    [Parameter(Mandatory = $true)][string]$PartitionDirectory,
    [string]$OutputDirectory = "",
    [ValidateSet('surface', 'legacy')][string]$Method = 'surface',
    [ValidateSet('cuda', 'cpu')][string]$ProjectionBackend = 'cuda',
    [ValidateRange(0.0, 1000000000.0)][double]$TargetEdgeLength = 0,
    [ValidateRange(0.0, 1000000000.0)][double]$MaxDeviation,
    [ValidateRange(1, 10000000)][int]$SampleCount = 2000,
    [ValidateRange(1, 10000)][int]$SplitPasses = 128,
    [ValidateRange(0, 10000)][int]$CollapsePasses = 12,
    [ValidateRange(0, 10000)][int]$FlipPasses = 8,
    [ValidateRange(0, 10000)][int]$RelaxIterations = 3,
    [ValidateRange(1, 10000000)][int]$BatchFaceLimit = 75000,
    [ValidateRange(0.0, 180.0)][double]$MaxNormalDeviationDegrees = 10.0,
    [switch]$TraceSurfaceValidation,
    [switch]$FullOutput,
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$moduleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $moduleRoot
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "PaMO Python was not found at $PythonExecutable. Supply -PythonExecutable with the configured PaMO environment."
}
if (-not (Test-Path -LiteralPath $PartitionDirectory)) {
    throw "Partition input does not exist: $PartitionDirectory"
}
$remeshArguments = @((Join-Path $moduleRoot 'remesh_partition.py'),
    [System.IO.Path]::GetFullPath($PartitionDirectory),
    '--sample-count', $SampleCount, '--split-passes', $SplitPasses,
    '--collapse-passes', $CollapsePasses, '--flip-passes', $FlipPasses,
    '--relax-iterations', $RelaxIterations, '--batch-face-limit', $BatchFaceLimit,
    '--max-normal-deviation-degrees',
    $MaxNormalDeviationDegrees.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture),
    '--method', $Method, '--projection-backend', $ProjectionBackend)
if ($TraceSurfaceValidation) {
    $remeshArguments += '--trace-surface-validation'
}
if ($FullOutput) {
    $remeshArguments += '--full-output'
}
if (-not [string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $remeshArguments += @('--output', [System.IO.Path]::GetFullPath($OutputDirectory))
}
if ($TargetEdgeLength -gt 0) {
    $remeshArguments += @('--target-edge-length',
        $TargetEdgeLength.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture))
}
if ($PSBoundParameters.ContainsKey('MaxDeviation')) {
    $remeshArguments += @('--max-deviation',
        $MaxDeviation.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture))
}
& $PythonExecutable @remeshArguments
if ($LASTEXITCODE -ne 0) { throw "PaMO partition remeshing failed (exit $LASTEXITCODE)." }
