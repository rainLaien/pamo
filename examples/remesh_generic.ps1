param(
    [string]$InputStl = '',
    [string]$OutputDirectory = '',
    [double]$TargetEdgeLength = 6,
    [double]$MaximumDeviation = 0.03,
    [double]$FeatureAngle = 45,
    [ValidateRange(1,30)][int]$Iterations = 5,
    [switch]$SkipBuild,
    [switch]$AnalyticGuides
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $InputStl) { $InputStl = Join-Path $PSScriptRoot '2.stl' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $PSScriptRoot ('generic_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
if (-not $SkipBuild) {
    & cmake --build (Join-Path $root 'cad_mesh/win') --config Release --target cad_mesh_generic --parallel 8
    if ($LASTEXITCODE -ne 0) { throw 'Generic remesh build failed' }
}
$culture = [Globalization.CultureInfo]::InvariantCulture
$arguments = @($InputStl, $OutputDirectory, '--target', $TargetEdgeLength.ToString($culture),
    '--deviation', $MaximumDeviation.ToString($culture), '--feature-angle', $FeatureAngle.ToString($culture), '--iterations', [string]$Iterations)
$ErrorActionPreference = 'Continue'
if ($AnalyticGuides) { $arguments += @('--analytic-guides', '1') }
& (Join-Path $root 'cad_mesh/win/Release/cad_mesh_generic.exe') @arguments 2>&1 |
    ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $OutputDirectory 'generic.log')
$exit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exit -ne 0) { throw "Generic remesh failed: $exit" }
Write-Host "Result: $OutputDirectory/generic_result.ply"
