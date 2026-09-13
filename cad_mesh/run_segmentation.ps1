param(
    [string]$InputStl = "",
    [string]$OutputDirectory = "",
    [string]$VcglibRoot = "d:\openSourceInstall\vcglib",
    [ValidateRange(0.0, 1.0)][double]$StrongBoundary = 0.72,
    [ValidateRange(0.0, 1.0)][double]$WeakBoundary = 0.42,
    [ValidateRange(1, 6)][int]$CurvatureRings = 2,
    [ValidateRange(0.01, 100)][double]$FitToleranceRatio = 3,
    [ValidateRange(0.01, 89)][double]$NormalAngleDegrees = 8.0214091318,
    [ValidateRange(0.01, 89)][double]$SharpAngleDegrees = 37.2422566835,
    [ValidateSet('auto', 'cpu', 'cuda')][string]$AnalyticSeedBackend = 'cuda',
    [switch]$Legacy,
    [switch]$Remesh,
    [ValidateSet('surface', 'legacy')][string]$RemeshMethod = 'surface',
    [ValidateRange(0.0, 1000000000.0)][double]$RemeshEdgeLength = 0,
    [ValidateRange(1, 10000000)][int]$RemeshSampleCount = 2000,
    [string]$RemeshOutputDirectory = "",
    [string]$PythonExecutable = "",
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
$moduleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $moduleRoot
if ([string]::IsNullOrWhiteSpace($InputStl)) {
    $InputStl = Join-Path $projectRoot "examples\Unnamed-Body.stl"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $moduleRoot "debug\Unnamed-Body"
}
$InputStl = [System.IO.Path]::GetFullPath($InputStl)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$buildDirectory = Join-Path $moduleRoot "build_mingw_release"
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) { $cmake = "C:\Program Files\CMake\bin\cmake.exe" }
if (-not (Test-Path -LiteralPath $cmake)) { throw "CMake was not found." }
if (-not (Test-Path -LiteralPath (Join-Path $VcglibRoot "vcg\complex\complex.h"))) { throw "VCGLib was not found at $VcglibRoot" }

$compiler = Get-ChildItem "D:\MinGW-W64*\*\bin\g++.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $compiler) { throw "g++.exe was not found. Run this script from a Visual Studio developer shell or set up MinGW." }
$env:PATH = "$($compiler.DirectoryName);$env:PATH"

& $cmake -S $moduleRoot -B $buildDirectory -G Ninja "-DVCGLIB_ROOT=$($VcglibRoot -replace '\\','/')" "-DCMAKE_CXX_COMPILER=$($compiler.FullName -replace '\\','/')" -DCADMESH_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }
& $cmake --build $buildDirectory --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed." }
if ($RunTests) {
    & (Join-Path $buildDirectory "cad_mesh_tests.exe")
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
}
$segmentArguments = @($InputStl, $OutputDirectory, '--fit-tolerance-ratio', $FitToleranceRatio,
    '--normal-angle-deg', $NormalAngleDegrees, '--sharp-angle-deg', $SharpAngleDegrees,
    '--analytic-seed-backend', $AnalyticSeedBackend)
if ($Legacy) { $segmentArguments += @('--legacy', '--strong', $StrongBoundary, '--weak', $WeakBoundary, '--rings', $CurvatureRings) }
& (Join-Path $buildDirectory "cad_mesh_segment.exe") @segmentArguments
if ($LASTEXITCODE -ne 0) { throw "Segmentation failed." }
if ($Remesh) {
    & (Join-Path $moduleRoot 'run_remesh.ps1') `
        -PartitionDirectory $OutputDirectory `
        -OutputDirectory $RemeshOutputDirectory `
        -TargetEdgeLength $RemeshEdgeLength `
        -SampleCount $RemeshSampleCount `
        -Method $RemeshMethod `
        -PythonExecutable $PythonExecutable
}
