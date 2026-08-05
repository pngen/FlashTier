<#
.SYNOPSIS
Configures and builds FlashTier with a CMake preset.
.EXAMPLE
.\scripts\build.ps1 -Preset windows-cuda-release
.\scripts\build.ps1 -Preset windows-cpu-release -Config Release
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Preset,
    [string]$Config = "Release",
    [switch]$SkipConfigure
)

$ErrorActionPreference = 'Stop'

if (-not $SkipConfigure) {
    Write-Host ">>> cmake --preset $Preset"
    & cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed with exit code $LASTEXITCODE"; exit $LASTEXITCODE }
}

$buildDir = Join-Path $PSScriptRoot "..\build\$Preset"
Write-Host ">>> cmake --build `"$buildDir`" --config $Config --parallel"
& cmake --build "$buildDir" --config $Config --parallel
if ($LASTEXITCODE -ne 0) { Write-Error "CMake build failed with exit code $LASTEXITCODE"; exit $LASTEXITCODE }

Write-Host "Build succeeded." -ForegroundColor Green
exit 0
