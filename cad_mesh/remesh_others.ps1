param(
    [string]$PartitionDirectory = '',
    [string]$OutputDirectory = '',
    [ValidateRange(1, 128)][int]$AnalyticWorkers = 4,
    [ValidateRange(0.000001, 1000000000.0)][double]$TargetEdgeLength = 6.0,
    [ValidateRange(0.0, 1000000000.0)][double]$MaxDeviation = 0.1,
    [ValidateRange(0.000001, 180.0)][double]$MaxNormalDeviationDegrees = 10.0,
    [switch]$PatchDetails,
    [ValidateRange(0.0, 180.0)][double]$GenericFeatureAngleDegrees = 10.0,
    [ValidateRange(1, 100)][int]$GenericRemeshIterations = 5,
    [ValidateRange(1, 128)][int]$GenericRemeshWorkers = 20
)
$ErrorActionPreference = 'Stop'
# Uses the saved partition directly: no STL segmentation or fitting stage.
$options = @{
    PartitionDirectory = $PartitionDirectory
    OutputDirectory = $OutputDirectory
    AnalyticWorkers = $AnalyticWorkers
    TargetEdgeLength = $TargetEdgeLength
    MaxDeviation = $MaxDeviation
    MaxNormalDeviationDegrees = $MaxNormalDeviationDegrees
    GenericFeatureAngleDegrees = $GenericFeatureAngleDegrees
    GenericRemeshIterations = $GenericRemeshIterations
    GenericRemeshWorkers = $GenericRemeshWorkers
    OtherFeaturesOnly = $true
    PatchDetails = $PatchDetails.IsPresent
}
Write-Host '[remesh] Others only: Cone, Sphere, Torus, Freeform; Plane/Cylinder interiors retained.'
Write-Host '[remesh] PLY remeshed: 0=not selected, 1=construction failed, 2=processed, 3=preserved input.'
$timer = [Diagnostics.Stopwatch]::StartNew()
try {
    & (Join-Path $PSScriptRoot 'remesh_saved.ps1') @options
} finally {
    $timer.Stop()
    Write-Host ('[remesh] Saved-partition others time: {0:F3} s (packaging, loading, remesh and export; no partition fitting)' -f $timer.Elapsed.TotalSeconds)
}
