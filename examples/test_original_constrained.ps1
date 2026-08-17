param(
    [switch]$EnableQualityOptimization,
    [ValidateRange(0.0, 180.0)]
    [double]$FeatureAngle = 15.0,
    [ValidateRange(0.000001, [double]::MaxValue)]
    [double]$MaxEdgeLength = 10.0
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$inputPath = Join-Path $projectRoot "examples\222.stl"
$outputDirectory = Join-Path $projectRoot "examples\test_outputs"
$outputPath = Join-Path $outputDirectory "222_constrained.stl"
$logPath = Join-Path $outputDirectory "222_constrained.log"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment not found: $pythonPath"
}

if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "Input mesh not found: $inputPath"
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

Push-Location $projectRoot
try {
    # Million-edge meshes are extremely slow in the current Python edge-flip
    # implementation. The default performs safe constrained refinement only.
    # Targeted fan cleanup is safe for this large mesh: it only scans edges
    # around abnormally high-valence vertices instead of every mesh edge.
    $flipPasses = 0
    $qualityIterations = if ($EnableQualityOptimization) { 1 } else { 0 }
    $qualityFlipPasses = if ($EnableQualityOptimization) { 1 } else { 0 }

    $arguments = @(
        "-u"
        ".\example.py"
        "--input", $inputPath
        "--output", $outputPath
        "--original-constrained-remesh"
        "--sdf-mode", "exact"
        "--constraint-feature-angle", "$FeatureAngle"
        "--constraint-max-edge-length", "$MaxEdgeLength"
        "--coplanar-angle-tolerance", "0.1"
        "--constraint-flip-passes", "$flipPasses"
        "--constraint-flip-minimum-valence", "12"
        "--constraint-flip-maximum-candidate-quality", "0.1"
        "--constraint-planar-fan-minimum-valence", "30"
        "--constraint-planar-annulus-minimum-faces", "20"
        "--constraint-cylinder-minimum-faces", "20"
        "--constraint-cylinder-radius-tolerance", "0.001"
        "--constraint-cylinder-target-edge-ratio", "1.0"
        "--constraint-trimmed-cylinder-minimum-faces", "20"
        "--constraint-trimmed-cylinder-radius-tolerance", "0.01"
        "--constraint-trimmed-cylinder-normal-tolerance", "0.08"
        "--constraint-partial-cylinder-minimum-faces", "20"
        "--constraint-partial-cylinder-radius-tolerance", "0.002"
        "--constraint-partial-cylinder-normal-tolerance", "0.02"
        "--constraint-partial-cylinder-minimum-angle", "30"
        "--constraint-rounded-fillet-minimum-faces", "12"
        "--constraint-rounded-fillet-minimum-curvature", "0.2"
        "--constraint-rounded-fillet-maximum-source-quality", "0.15"
        "--constraint-rounded-fillet-target-edge-ratio", "2.0"
        "--constraint-rounded-fillet-minimum-triangle-angle", "28.0"
        "--constraint-planar-region-minimum-faces", "20"
        "--constraint-quality-iterations", "$qualityIterations"
        "--constraint-quality-step", "0.4"
        "--constraint-quality-flip-passes", "$qualityFlipPasses"
    )
    & $pythonPath @arguments 2>&1 | Tee-Object -FilePath $logPath
    $pythonExitCode = $LASTEXITCODE

    if ($pythonExitCode -ne 0) {
        throw "Mesh optimization failed with exit code $pythonExitCode"
    }
}
finally {
    Pop-Location
}

Write-Host "Optimized mesh written to: $outputPath"
Write-Host "Timing log written to: $logPath"
