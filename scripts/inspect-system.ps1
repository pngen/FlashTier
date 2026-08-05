<#
.SYNOPSIS
Inspects the machine for FlashTier prerequisites: CMake, MSVC toolchain,
CUDA toolkit, NVIDIA GPU, system RAM, and storage.
.EXAMPLE
.\scripts\inspect-system.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$script:failed = $false

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host "==== $Title ====" -ForegroundColor Cyan
}

function Write-Ok([string]$Msg)  { Write-Host "  [OK]   $Msg" -ForegroundColor Green }
function Write-Warn([string]$Msg){ Write-Host "  [WARN] $Msg" -ForegroundColor Yellow; $script:failed = $true }
function Write-Info([string]$Msg){ Write-Host "  [INFO] $Msg" }

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$cs = Get-CimInstance Win32_ComputerSystem

Write-Section "Operating System"
Write-Info "$($os.Caption) $($os.Version) ($($os.OSArchitecture))"

Write-Section "CPU"
Write-Info "$($cpu.Name) - $($cpu.NumberOfCores) cores / $($cpu.NumberOfLogicalProcessors) logical"

Write-Section "System Memory"
$totalRamGiB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
$freeRamGiB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
Write-Info "Total: $totalRamGiB GiB  Free: $freeRamGiB GiB"

Write-Section "CMake"
try {
    $cmake = & cmake --version 2>&1 | Select-Object -First 1
    if ($LASTEXITCODE -ne 0) { throw "cmake failed" }
    Write-Ok $cmake
} catch { Write-Warn "CMake not found on PATH" }

Write-Section "MSVC Toolchain"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $instances = & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    foreach ($inst in $instances) {
        $vcvars = Join-Path $inst "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvars) { Write-Ok "Visual Studio at $inst" }
    }
    if (-not $instances) { Write-Warn "No Visual Studio with VC++ x64 tools found" }
} else { Write-Warn "vswhere not found; cannot detect Visual Studio" }

Write-Section "NVIDIA GPU"
try {
    $smi = & nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>&1
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed" }
    foreach ($line in $smi) { Write-Ok $line }
} catch { Write-Warn "nvidia-smi not found; no NVIDIA GPU detected" }

Write-Section "CUDA Toolkit"
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $cudaRoot) {
    $versions = Get-ChildItem $cudaRoot -Directory | Select-Object -ExpandProperty Name
    if ($versions) {
        Write-Ok "Toolkits found: $($versions -join ', ')"
        $nvcc = Join-Path $cudaRoot (Join-Path $versions[0] "bin\nvcc.exe")
        if (Test-Path $nvcc) {
            & $nvcc --version 2>&1 | ForEach-Object { Write-Info $_ }
        }
    } else { Write-Warn "CUDA directory exists but no toolkit versions found" }
} else { Write-Warn "CUDA toolkit not found at $cudaRoot" }

Write-Section "Storage"
Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    $freeGiB = [math]::Round($_.Free / 1GB, 1)
    Write-Info "Drive $($_.Name): $freeGiB GiB free"
}

Write-Section "Result"
if ($script:failed) {
    Write-Host "Inspection completed with warnings (see above)." -ForegroundColor Yellow
    exit 1
}
Write-Host "Inspection completed successfully." -ForegroundColor Green
exit 0
