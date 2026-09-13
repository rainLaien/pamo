param(
    [switch]$WallThickness,
    [ValidateRange(1, 128)][int]$ThicknessWorkers = 20,
    [ValidateRange(0.0, 1000000000.0)][double]$ThicknessMinimum = 0.01,
    [ValidateRange(0.0, 180.0)][double]$ThicknessContactAngleDegrees = 0.0,
    [string]$InputStl = '',
    [string]$OutputDirectory = '',
    [ValidateRange(0.000001, 1000000000.0)][double]$TargetEdgeLength = 6.0,
    [ValidateRange(0.0, 1000000000.0)][double]$MaxDeviation = 0.1,
    [ValidateRange(0.000001, 180.0)][double]$MaxNormalDeviationDegrees = 10.0,
    [ValidateRange(1, 128)][int]$AnalyticWorkers = 8,
    [ValidateRange(1, 128)][int]$GenericRemeshWorkers = 20,
    [switch]$PatchDetails
)

$ErrorActionPreference = 'Stop'
# Use the same end-to-end pipeline and output handling as the CUDA entry point.
# --cpu in the executable also disables CUDA in secondary model fitting.
$arguments = @{
    InputStl = $InputStl
    OutputDirectory = $OutputDirectory
    TargetEdgeLength = $TargetEdgeLength
    MaxDeviation = $MaxDeviation
    MaxNormalDeviationDegrees = $MaxNormalDeviationDegrees
    AnalyticWorkers = $AnalyticWorkers
    GenericRemeshWorkers = $GenericRemeshWorkers
    WallThickness = $WallThickness.IsPresent
    ThicknessWorkers = $ThicknessWorkers
    ThicknessMinimum = $ThicknessMinimum
    ThicknessContactAngleDegrees = $ThicknessContactAngleDegrees
    PatchDetails = $PatchDetails
    Cpu = $true
}
& (Join-Path $PSScriptRoot 'cuda.ps1') @arguments
