<#
.SYNOPSIS
Runs the FlashTier test suite via CTest.
.EXAMPLE
.\scripts\test.ps1 -Preset windows-cuda-release -Config Release
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Preset,
    [string]$Config = "Release"
)

$ErrorActionPreference = 'Stop'

$buildDir = Join-Path $PSScriptRoot "..\build\$Preset"
if (-not (Test-Path $buildDir)) {
    Write-Error "Build directory '$buildDir' does not exist. Run build.ps1 first."
    exit 2
}

Write-Host ">>> ctest --test-dir `"$buildDir`" -C $Config --output-on-failure"
& ctest --test-dir "$buildDir" -C $Config --output-on-failure
if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed with exit code $LASTEXITCODE"; exit $LASTEXITCODE }

Write-Host "All tests passed." -ForegroundColor Green
exit 0
