# Build altd_gpu.dll (the GPU simulator parts) for NVIDIA GPUs into $OUT.
# CUDA_ARCH is native (the default: the GPUs of this machine; without one nvcc warns and uses
# sm_75) or one or more sm_ targets separated by commas or spaces. A list also embeds PTX for its
# newest target, which the driver compiles at load time for newer GPUs.
# -fmad=false: nvcc otherwise fuses a*b+c into FMA, which rounds once. Keep the default
# -prec-div=true -prec-sqrt=true -ftz=false and never use -use_fast_math.
# -AllowUnsupportedCompiler (or CUDA_ALLOW_UNSUPPORTED_COMPILER=1) passes -allow-unsupported-compiler
# to nvcc for a Visual Studio newer than the CUDA toolkit supports. NVIDIA has not validated that
# combination; run the CPU/GPU checks in gpu/README.md before trusting the result.
param(
    [string]$Out = $(if ($env:OUT) { $env:OUT } else { Join-Path $PSScriptRoot "..\target\gpu" }),
    [string]$Arch = $(if ($env:CUDA_ARCH) { $env:CUDA_ARCH } else { "native" }),
    [switch]$AllowUnsupportedCompiler = ($env:CUDA_ALLOW_UNSUPPORTED_COMPILER -eq "1")
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path

# The py launcher first: python3 on Windows is often the Microsoft Store placeholder.
$tables = Join-Path $PSScriptRoot "gen_tables.py"
if (Get-Command py -ErrorAction SilentlyContinue) { py -3 $tables --check } else { python $tables --check }
if ($LASTEXITCODE -ne 0) { throw "gen_tables.py --check failed" }

if ($Arch -eq "native") {
    $archFlags = "-arch=native"
} else {
    $targets = @($Arch -split '[,\s]+' | Where-Object { $_ })
    if (-not $targets) { throw "CUDA_ARCH lists no targets" }
    $bad = @($targets | Where-Object { $_ -notmatch '^sm_\d+$' })
    if ($bad) { throw "CUDA_ARCH target $($bad[0]) is not sm_NN (or the whole value native)" }
    $numbers = @($targets | ForEach-Object { [int]($_ -replace '^sm_', '') })
    $newest = ($numbers | Measure-Object -Maximum).Maximum
    $archFlags = (@($numbers | ForEach-Object { "-gencode=arch=compute_$_,code=sm_$_" }) +
        "-gencode=arch=compute_$newest,code=compute_$newest") -join " "
}

$unsupported = ""
if ($AllowUnsupportedCompiler) {
    Write-Warning "building with -allow-unsupported-compiler: NVIDIA has not validated this Visual Studio with this CUDA toolkit"
    $unsupported = "-allow-unsupported-compiler"
}

# nvcc needs the MSVC host compiler environment; load it unless cl.exe is already on PATH.
$src = Join-Path $PSScriptRoot "sim\altd_gpu.hip"
$dll = Join-Path $Out "altd_gpu.dll"
$prefix = ""
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vw = Join-Path ([Environment]::GetFolderPath("ProgramFilesX86")) "Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vw -products * -latest -property installationPath
    $prefix = "`"$(Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat')`" >nul && "
}
$cmd = "${prefix}nvcc -x cu -O3 -std=c++20 -fmad=false $unsupported $archFlags -shared -Xcompiler /MD -o `"$dll`" `"$src`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "nvcc failed" }
Write-Host "built $dll for $Arch"
