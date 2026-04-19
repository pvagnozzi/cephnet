# run_prepare.ps1
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: Windows wrapper — downloads/generates both the training dataset and the
#              Aariz verification dataset inside the cephnet-data Docker named volume.

param(
    [string]$Config     = "configs/training/quick_test.yaml",
    [switch]$Synthetic,   # Skip download, generate synthetic data
    [switch]$Force,       # Overwrite existing datasets in the volume
    [int]$NTrain        = 80,
    [int]$NVal          = 20,
    [int]$NTest         = 20,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "../..")

. (Join-Path $ScriptDir "_common.ps1")
Ensure-ArtifactDirs -RepoRoot $RepoRoot
$ComposeFile = Get-ComposeFile -RepoRoot $RepoRoot

# --- Step 1: Prepare training dataset ---
$CmdArgs = @("--config", $Config, "--n-train", $NTrain, "--n-val", $NVal, "--n-test", $NTest)
if ($Synthetic) { $CmdArgs += "--synthetic" }
if ($Force)     { $CmdArgs += "--force" }
$CmdArgs += $ExtraArgs

Write-Host "Step 1/2 — Preparing training dataset (config: $Config)" -ForegroundColor Cyan
if ($Synthetic) {
    Write-Host "   Mode: SYNTHETIC -- data written to Docker volume: cephnet-data" -ForegroundColor Yellow
} else {
    Write-Host "   Mode: download (fallback to synthetic) -- data written to Docker volume: cephnet-data" -ForegroundColor Green
}

docker compose `
    -f $ComposeFile `
    --project-directory $RepoRoot `
    run --rm trainer `
    uv run python scripts/python/prepare_dataset.py @CmdArgs

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# --- Step 2: Prepare Aariz verification dataset ---
$AarizArgs = @(
    "--config", "configs/training/verify_aariz.yaml",
    "--n-train", 0, "--n-val", 0, "--n-test", 50
)
if ($Synthetic) { $AarizArgs += "--synthetic" }
if ($Force)     { $AarizArgs += "--force" }

Write-Host "Step 2/2 — Preparing Aariz verification dataset" -ForegroundColor Cyan

docker compose `
    -f $ComposeFile `
    --project-directory $RepoRoot `
    run --rm trainer `
    uv run python scripts/python/prepare_dataset.py @AarizArgs

exit $LASTEXITCODE

