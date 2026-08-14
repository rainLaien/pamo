$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$inputPath = Join-Path $projectRoot "examples\Unnamed-Body.stl"
$outputDirectory = Join-Path $projectRoot "examples\test_outputs"
$outputPath = Join-Path $outputDirectory "Unnamed-Body_original_constrained.stl"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment not found: $pythonPath"
}

if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "Input mesh not found: $inputPath"
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

Push-Location $projectRoot
try {
    & $pythonPath ".\example.py" `
        --input $inputPath `
        --output $outputPath `
        --original-constrained-remesh `
        --sdf-mode exact `
        --constraint-max-edge-length 10 `
        --constraint-feature-angle 5 `
        --constraint-max-splits 5000 `
        --coplanar-angle-tolerance 0.1 `
        --constraint-flip-passes 8 `
        --constraint-quality-iterations 20 `
        --constraint-quality-step 0.4 `
        --constraint-quality-flip-passes 12

    if ($LASTEXITCODE -ne 0) {
        throw "Mesh optimization failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Optimized mesh written to: $outputPath"
