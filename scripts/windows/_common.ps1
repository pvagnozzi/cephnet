# _common.ps1
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-19
# Last modified: 2026-04-19
# Description: Shared helpers for cephnet Windows wrapper scripts.

# Ensure all host-side artifact directories exist before Docker bind-mounts them.
# Data is intentionally NOT created here — it lives in the cephnet-data named volume.
function Ensure-ArtifactDirs {
    param([string]$RepoRoot)
    $dirs = @(
        "models\checkpoints",
        "models\exports",
        "models\artifacts",
        "reports\metrics",
        "reports\plots",
        "reports\verification",
        "reports\logs"
    )
    foreach ($d in $dirs) {
        $full = Join-Path $RepoRoot $d
        if (-not (Test-Path $full)) {
            New-Item -ItemType Directory -Path $full -Force | Out-Null
        }
    }
}

# Resolve the correct compose file based on -Gpu flag.
function Get-ComposeFile {
    param([string]$RepoRoot, [switch]$Gpu)
    if ($Gpu) {
        return Join-Path $RepoRoot "containers\compose\docker-compose.gpu.yml"
    }
    return Join-Path $RepoRoot "containers\compose\docker-compose.cpu.yml"
}
