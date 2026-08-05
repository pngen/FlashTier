<#
.SYNOPSIS
Runs FlashTier benchmarks with safe default sizes.
.EXAMPLE
.\scripts\benchmark.ps1 -Preset windows-cuda-release -Config Release
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Preset,
    [string]$Config = "Release",
    [switch]$TiersOnly
)

$ErrorActionPreference = 'Stop'

$buildDir = Join-Path $PSScriptRoot "..\build\$Preset"
$exe = Join-Path $buildDir "bin\flashtier.exe"
if (-not (Test-Path $exe)) {
    $alt = Join-Path $buildDir "flashtier.exe"
    if (-not (Test-Path $alt)) {
        Write-Error "flashtier executable not found under '$buildDir'. Run build.ps1 first."
        exit 2
    }
    $exe = $alt
}

function Invoke-Flashtier([string[]]$Args, [string]$Label) {
    Write-Host ""
    Write-Host "===== $Label =====" -ForegroundColor Cyan
    & $exe @Args
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Label failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Invoke-Flashtier @("benchmark", "tiers", "--strict") "benchmark tiers"
Invoke-Flashtier @("benchmark", "oversubscription", "--strict") "benchmark oversubscription"
Invoke-Flashtier @("benchmark", "prefetch", "--strict") "benchmark prefetch"

if (-not $TiersOnly) {
    Invoke-Flashtier @("benchmark", "sparse-experts", "--strict") "benchmark sparse-experts"
}

Write-Host ""
Write-Host "Benchmarks completed." -ForegroundColor Green
exit 0
