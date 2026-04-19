# run_verify.ps1
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: Windows wrapper — cross-dataset verification. Reads data from the
#              cephnet-data volume; writes verification reports to host-mounted reports/.

param(
    [string]$Config              = "configs/training/quick_test.yaml",
    [string]$VerificationDataset = "aariz",
    [string]$VerificationRoot    = "/data/processed/aariz",
    [switch]$Gpu,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "../..")

. (Join-Path $ScriptDir "_common.ps1")
Ensure-ArtifactDirs -RepoRoot $RepoRoot
$ComposeFile = Get-ComposeFile -RepoRoot $RepoRoot -Gpu:$Gpu

Write-Host "Cross-dataset verification (config: $Config, dataset: $VerificationDataset)" -ForegroundColor Cyan
Write-Host "   Reports -> $RepoRoot\reports\verification" -ForegroundColor Green

docker compose `
    -f $ComposeFile `
    --project-directory $RepoRoot `
    run --rm trainer `
    uv run python scripts/python/verify_cross_dataset.py `
    --config $Config `
    --verification-dataset $VerificationDataset `
    --verification-root $VerificationRoot `
    @ExtraArgs

exit $LASTEXITCODE

