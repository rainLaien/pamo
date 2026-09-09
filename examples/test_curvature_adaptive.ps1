param(
    [ValidateRange(1, [int]::MaxValue)]
    [int]$SampleCount = 8000,

    [ValidateRange(0.000001, [double]::MaxValue)]
    [double]$PoissonRadius = 8.0,

    [ValidateRange(0.000001, [double]::MaxValue)]
    [double]$CurvatureTolerance = 1.0,

    [ValidateRange(0.000001, [double]::MaxValue)]
    [double]$MinimumEdgeLength = 5.0,

    [ValidateRange(0.000001, [double]::MaxValue)]
    [double]$MaximumEdgeLength = 30.0,

    [ValidateRange(0.000001, 179.999999)]
    [double]$FeatureAngle = 30.0,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$SplitPasses = 24,

    [ValidateRange(0, [int]::MaxValue)]
    [int]$CollapsePasses = 4,

    [ValidateRange(0, [int]::MaxValue)]
    [int]$FlipPasses = 4,

    [ValidateRange(0, [int]::MaxValue)]
    [int]$RelaxIterations = 1
)

$ErrorActionPreference = "Stop"

if ($MinimumEdgeLength -gt $MaximumEdgeLength) {
    throw "MinimumEdgeLength must not exceed MaximumEdgeLength."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$inputPath = Join-Path $projectRoot "examples\111.stl"
$outputDirectory = Join-Path $projectRoot "examples\test_outputs"
$outputPath = Join-Path $outputDirectory "111_curvature_adaptive_debug.stl"
$logPath = Join-Path $outputDirectory "111_curvature_adaptive_debug.log"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "Input mesh not found: $inputPath"
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$arguments = @(
    "-u"
    ".\example.py"
    "--input", $inputPath
    "--output", $outputPath
    "--surface-sample-remesh"
    "--surface-curvature-adaptive"
    "--surface-sample-count", "$SampleCount"
    "--surface-poisson-radius", "$PoissonRadius"
    "--surface-curvature-tolerance", "$CurvatureTolerance"
    "--surface-adaptive-min-edge-length", "$MinimumEdgeLength"
    "--surface-adaptive-max-edge-length", "$MaximumEdgeLength"
    "--feature-edge-angle", "$FeatureAngle"
    "--surface-split-passes", "$SplitPasses"
    "--surface-collapse-passes", "$CollapsePasses"
    "--surface-flip-passes", "$FlipPasses"
    "--surface-relax-iterations", "$RelaxIterations"
)

Write-Host "Input:  $inputPath"
Write-Host "Output: $outputPath"
Write-Host "Log:    $logPath"
Write-Host (
    "Adaptive target: min={0}, max={1}, tolerance={2}, Poisson={3}" -f
    $MinimumEdgeLength, $MaximumEdgeLength, $CurvatureTolerance, $PoissonRadius
)

Push-Location $projectRoot
try {
    & $pythonPath @arguments 2>&1 | Tee-Object -FilePath $logPath
    $pythonExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($pythonExitCode -ne 0) {
    throw "Curvature-adaptive remeshing failed with exit code $pythonExitCode."
}
if (-not (Test-Path -LiteralPath $outputPath)) {
    throw "Remeshing completed without creating the expected output file."
}

$outputFile = Get-Item -LiteralPath $outputPath
Write-Host "Remesh command completed."
Write-Host "Output size: $([math]::Round($outputFile.Length / 1MB, 2)) MiB"
Write-Host "Output mesh: $outputPath"
Write-Host "Timing log:  $logPath"

$validationCode = @'
import sys
import numpy as np
import trimesh

input_mesh = trimesh.load(sys.argv[1], force="mesh", process=True)
output_mesh = trimesh.load(sys.argv[2], force="mesh", process=True)
input_components = len(input_mesh.split(only_watertight=False))
output_components = len(output_mesh.split(only_watertight=False))
finite = bool(np.isfinite(output_mesh.vertices).all())

print("Validation:")
print(f"  vertices={len(output_mesh.vertices)}, faces={len(output_mesh.faces)}")
print(f"  finite={finite}, winding_consistent={output_mesh.is_winding_consistent}")
print(f"  watertight={output_mesh.is_watertight}")
print(f"  connected_components={input_components} -> {output_components}")

valid = (
    finite
    and output_mesh.is_winding_consistent
    and output_components <= input_components
)
raise SystemExit(0 if valid else 2)
'@

& $pythonPath -c $validationCode $inputPath $outputPath
$validationExitCode = $LASTEXITCODE
if ($validationExitCode -ne 0) {
    Write-Warning (
        "The remesh process exited normally, but topology validation failed. " +
        "Treat this output as a debug artifact, not a final mesh."
    )
}
else {
    Write-Host "Topology validation passed."
}
