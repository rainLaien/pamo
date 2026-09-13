param(
    [switch]$WallThickness,
    [ValidateRange(1, 128)][int]$ThicknessWorkers = 20,
    [ValidateRange(0.0, 1000000000.0)][double]$ThicknessMinimum = 0.01,
    [ValidateRange(0.0, 180.0)][double]$ThicknessContactAngleDegrees = 0.0,
    [string]$PartitionDirectory = '',
    [string]$OutputDirectory = '',
    [ValidateRange(1, 128)][int]$AnalyticWorkers = 4,
    [ValidateRange(0.000001, 1000000000.0)][double]$TargetEdgeLength = 6.0,
    [ValidateRange(0.0, 1000000000.0)][double]$MaxDeviation = 0.1,
    [ValidateRange(0.000001, 180.0)][double]$MaxNormalDeviationDegrees = 10.0,
    [ValidateRange(0.0, 180.0)][double]$GenericFeatureAngleDegrees = 10.0,
    [ValidateRange(1, 100)][int]$GenericRemeshIterations = 5,
    [ValidateRange(1, 128)][int]$GenericRemeshWorkers = 20,
    [switch]$CpuAnalytic,
    [switch]$PatchDetails
)
$ErrorActionPreference = 'Stop'
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
    CpuAnalytic = $CpuAnalytic.IsPresent
    WallThickness = $WallThickness.IsPresent
    ThicknessWorkers = $ThicknessWorkers
    ThicknessMinimum = $ThicknessMinimum
    ThicknessContactAngleDegrees = $ThicknessContactAngleDegrees
    PatchDetails = $PatchDetails.IsPresent
}
Write-Host '[remesh] All surface types enabled, including original Plane and Cylinder patches.'
Write-Host '[remesh] Loading saved partition; Others retain length-compliant faces and repartition the remainder.'
$timer = [Diagnostics.Stopwatch]::StartNew()
try {
    & (Join-Path $PSScriptRoot 'remesh_saved.ps1') @options
} finally {
    $timer.Stop()
    Write-Host ('[remesh] Saved-partition all-types time: {0:F3} s (packaging, local repartition, remesh and export)' -f $timer.Elapsed.TotalSeconds)
}
