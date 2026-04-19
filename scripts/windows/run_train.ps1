# run_train.ps1
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: Windows wrapper — runs cephnet training via Docker Compose.
#   Artifacts (checkpoints, ONNX exports) are written to host-mounted models/.
#   Reports (metrics, HTML) are written to host-mounted reports/.
#   Training data lives in the cephnet-data named Docker volume.

param(
    [string]$Config = "configs/training/quick_test.yaml",
    [switch]$Gpu,
    [switch]$Resume,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "../..")

. (Join-Path $ScriptDir "_common.ps1")
Ensure-ArtifactDirs -RepoRoot $RepoRoot
$ComposeFile = Get-ComposeFile -RepoRoot $RepoRoot -Gpu:$Gpu

$CmdArgs = @("--config", $Config)
if ($Resume) { $CmdArgs += "--resume" }
$CmdArgs += $ExtraArgs

Write-Host "Starting training (config: $Config)" -ForegroundColor Cyan
Write-Host "   Artifacts -> $RepoRoot\models" -ForegroundColor Green
Write-Host "   Reports   -> $RepoRoot\reports" -ForegroundColor Green
Write-Host "   Data      -> Docker volume: cephnet-data" -ForegroundColor Yellow

docker compose `
    -f $ComposeFile `
    --project-directory $RepoRoot `
    run --rm trainer `
    uv run python scripts/python/train_model.py @CmdArgs

exit $LASTEXITCODE

