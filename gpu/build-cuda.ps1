# Build altd_gpu.dll (the GPU simulator parts) for NVIDIA GPUs into $OUT.
# CUDA_ARCH lists one or more targets separated by commas or spaces, default sm_89 (RTX 40xx).
# -fmad=false: nvcc otherwise fuses a*b+c into FMA, which rounds once. Keep the default
# -prec-div=true -prec-sqrt=true -ftz=false and never use -use_fast_math.
param(
    [string]$Out = $(if ($env:OUT) { $env:OUT } else { Join-Path $PSScriptRoot "..\target\gpu" }),
    [string]$Arch = $(if ($env:CUDA_ARCH) { $env:CUDA_ARCH } else { "sm_89" })
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path

uv run $PSScriptRoot\gen_tables.py --check
if ($LASTEXITCODE -ne 0) { throw "gen_tables.py --check failed" }

$archFlags = ($Arch -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { "-gencode=arch=compute_$($_ -replace '^sm_',''),code=$_" }) -join " "
if (-not $archFlags) { throw "CUDA_ARCH lists no targets" }

# nvcc needs the MSVC host compiler environment; load it unless cl.exe is already on PATH.
$src = Join-Path $PSScriptRoot "sim\altd_gpu.hip"
$dll = Join-Path $Out "altd_gpu.dll"
$prefix = ""
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vw = Join-Path ([Environment]::GetFolderPath("ProgramFilesX86")) "Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vw -products * -latest -property installationPath
    $prefix = "`"$(Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat')`" >nul && "
}
$cmd = "${prefix}nvcc -x cu -O3 -std=c++20 -fmad=false -allow-unsupported-compiler $archFlags -shared -Xcompiler /MD -o `"$dll`" `"$src`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "nvcc failed" }
Write-Host "built $dll for $Arch"
