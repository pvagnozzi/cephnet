# run_validate.ps1
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: Windows wrapper — validates a trained model. Reads data from the
#              cephnet-data volume; writes reports to host-mounted reports/.

param(
    [string]$Config     = "configs/training/quick_test.yaml",
    [string]$Checkpoint = "",
    [switch]$Gpu,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "../..")

. (Join-Path $ScriptDir "_common.ps1")
Ensure-ArtifactDirs -RepoRoot $RepoRoot
$ComposeFile = Get-ComposeFile -RepoRoot $RepoRoot -Gpu:$Gpu

$CmdArgs = @("--config", $Config)
if ($Checkpoint) { $CmdArgs += @("--checkpoint", $Checkpoint) }
$CmdArgs += $ExtraArgs

Write-Host "Validating model (config: $Config)" -ForegroundColor Cyan
Write-Host "   Reports -> $RepoRoot\reports\metrics" -ForegroundColor Green

docker compose `
    -f $ComposeFile `
    --project-directory $RepoRoot `
    run --rm trainer `
    uv run python scripts/python/validate_model.py @CmdArgs

exit $LASTEXITCODE

