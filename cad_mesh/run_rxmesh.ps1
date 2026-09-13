param(
    [Parameter(Mandatory=$true)][string]$InputMesh,
    [string]$OutputDirectory = "",
    [double]$TargetEdgeRatio = 0.01,
    [double]$TargetEdgeLength = 0,
    [double]$MaximumDeviation = 0.03,
    [double]$NormalDegrees = 10,
    [double]$FeatureAngle = 45,
    [int]$Iterations = 5,
    [int]$MaxFaces = 4000000,
    [string]$VcglibRoot = "D:\opensource_install\vcglib",
    [string]$CudaVersion = "12.6",
    [string]$CudaArchitectures = "89",
    [string]$RxMeshSource = "",
    [switch]$PartitionSnapshot,
    [switch]$Segment,
    [switch]$SkipAudit,
    [switch]$SkipRelocation,
    [switch]$SkipCollapse,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
# Resolve against PowerShell's location; .NET's process directory can differ.
$InputMesh = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InputMesh)
if (-not (Test-Path -LiteralPath $InputMesh -PathType Leaf)) {
    throw "Input file not found: $InputMesh"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path (Split-Path -Parent $InputMesh) ("rxmesh_" + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$OutputDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
$buildDirectory = Join-Path $PSScriptRoot 'build_rxmesh'
$exe = Join-Path $buildDirectory 'rxmesh\Release\cad_mesh_rxmesh.exe'
if (-not $SkipBuild) {
    $configure = @('-S', $PSScriptRoot, '-B', $buildDirectory,
        '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', "cuda=$CudaVersion",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures", "-DVCGLIB_ROOT=$VcglibRoot",
        '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON',
        '-DCADMESH_BUILD_TESTS=OFF', '-DCADMESH_BUILD_RXMESH=ON')
    if ($RxMeshSource) { $configure += "-DFETCHCONTENT_SOURCE_DIR_RXMESH=$RxMeshSource" }
    & cmake @configure
    if ($LASTEXITCODE -ne 0) { throw 'RXMesh configuration failed.' }
    & cmake --build $buildDirectory --config Release --target cad_mesh_rxmesh --parallel 8
    if ($LASTEXITCODE -ne 0) { throw 'RXMesh build failed.' }
}
if (-not (Test-Path -LiteralPath $exe)) { throw "Executable not found: $exe" }
$culture = [Globalization.CultureInfo]::InvariantCulture
$runArgs = @($InputMesh, $OutputDirectory,
    '--target-edge-ratio', $TargetEdgeRatio.ToString('R',$culture),
    '--max-deviation', $MaximumDeviation.ToString('R',$culture),
    '--normal-degrees', $NormalDegrees.ToString('R',$culture),
    '--feature-angle', $FeatureAngle.ToString('R',$culture),
    '--iterations', $Iterations, '--max-faces', $MaxFaces)
if ($TargetEdgeLength -gt 0) { $runArgs += @('--target-edge-length', $TargetEdgeLength.ToString('R',$culture)) }
if ($PartitionSnapshot) { $runArgs += '--partition-snapshot' }
if ($Segment) { $runArgs += '--segment' }
if ($SkipAudit) { $runArgs += '--skip-audit' }
if ($SkipRelocation) { $runArgs += '--skip-relocation' }
if ($SkipCollapse) { $runArgs += '--skip-collapse' }
$timer = [Diagnostics.Stopwatch]::StartNew()
Write-Host "[RX] input=$InputMesh"
Write-Host "[RX] output=$OutputDirectory"
& $exe @runArgs
$runExitCode = $LASTEXITCODE
$timer.Stop()
Write-Host ("[RX] process_wall_seconds=" + $timer.Elapsed.TotalSeconds.ToString('F3', $culture))
if ($runExitCode -ne 0) { throw 'RXMesh remeshing failed; inspect the console and any candidate report.' }
