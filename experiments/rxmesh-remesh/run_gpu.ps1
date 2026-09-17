param(
    [string]$CudaVersion = "12.6",
    [string]$CudaArchitectures = "86;89",
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$build = Join-Path $root 'build_rx'
if (-not $SkipBuild) {
    $configure = @(
        '-S', $root, '-B', $build,
        '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', "cuda=$CudaVersion",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
        '-DCAD_ADAPTIVE_RXMESH=ON',
        '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON'
    )
    & cmake @configure
    if ($LASTEXITCODE -ne 0) { throw 'GPU remesh configure failed' }
    # Stale *.device-link.obj after RxMeshBackend.cu rebuild is LNK2001
    # (__fatbinwrap / __cudaRegisterLinkedBinary hash mismatch). Device-link
    # now lives in cad_adaptive_rx; still drop leftovers before MSBuild.
    Get-ChildItem -Recurse $build -Filter '*.device-link.obj' -ErrorAction SilentlyContinue |
        Remove-Item -Force
    & cmake --build $build --config Release --parallel 8
    if ($LASTEXITCODE -ne 0) { throw 'GPU remesh build failed' }
}
& ctest --test-dir $build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'GPU remesh tests failed' }
