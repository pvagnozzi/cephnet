# setup-branch-protection.ps1
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-19
# Last modified: 2026-04-19
# Description: Configures GitHub branch protection rules for GitFlow branches.
#              Requires 'gh' CLI authenticated with repo admin permissions.
#              Usage: .\scripts\windows\setup-branch-protection.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$REPO = "pvagnozzi/cephnet"

# ─── helpers ──────────────────────────────────────────────────────────────────

function Write-Ok   { param($msg) Write-Host "  ✅ $msg" -ForegroundColor Green  }
function Write-Warn { param($msg) Write-Host "  ⚠️  $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "  ❌ $msg" -ForegroundColor Red; exit 1 }
function Write-Step { param($msg) Write-Host "  $msg" }

function Test-Requirements {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-Fail "'gh' CLI not found. Install from https://cli.github.com"
    }
    $authStatus = gh auth status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Not authenticated with GitHub. Run: gh auth login"
    }
    Write-Ok "Requirements satisfied"
}

function Ensure-BranchExists {
    param([string]$Branch)

    $result = gh api "repos/$REPO/branches/$Branch" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Branch '$Branch' already exists"
        return
    }

    Write-Warn "Branch '$Branch' not found — creating from default branch"
    $defaultBranch = gh api "repos/$REPO" --jq '.default_branch'
    $sha = gh api "repos/$REPO/git/ref/heads/$defaultBranch" --jq '.object.sha'

    $body = @{ ref = "refs/heads/$Branch"; sha = $sha } | ConvertTo-Json
    $body | gh api "repos/$REPO/git/refs" --method POST --input - | Out-Null
    Write-Ok "Branch '$Branch' created from '$defaultBranch'"
}

function Protect-Branch {
    param([string]$Branch)

    Write-Step "Configuring protection for '$Branch'..."

    $payload = @{
        required_status_checks = @{
            strict   = $true
            contexts = @(
                "🔄 CI / 🏷️ Compute Version",
                "🔄 CI / 🎨 Lint & Type Check",
                "🔄 CI / 🧪 Unit Tests"
            )
        }
        enforce_admins = $false
        required_pull_request_reviews = @{
            required_approving_review_count = 1
            dismiss_stale_reviews           = $true
            require_code_owner_reviews      = $false
        }
        required_linear_history           = $false
        allow_force_pushes                = $false
        allow_deletions                   = $false
        required_conversation_resolution  = $true
    } | ConvertTo-Json -Depth 5

    $payload | gh api "repos/$REPO/branches/$Branch/protection" `
        --method PUT --input - | Out-Null

    Write-Ok "Branch '$Branch' protected (PR required, status checks required)"
}

function Protect-Pattern {
    param([string]$Pattern)

    Write-Step "Creating ruleset for pattern '$Pattern'..."

    $payload = @{
        name        = "Protect $Pattern"
        target      = "branch"
        enforcement = "active"
        conditions  = @{
            ref_name = @{
                include = @("refs/heads/$Pattern")
                exclude = @()
            }
        }
        rules = @(
            @{ type = "deletion" },
            @{ type = "non_fast_forward" },
            @{
                type       = "pull_request"
                parameters = @{
                    required_approving_review_count  = 1
                    dismiss_stale_reviews_on_push    = $true
                    require_code_owner_review        = $false
                    require_last_push_approval       = $false
                    required_review_thread_resolution = $true
                }
            },
            @{
                type       = "required_status_checks"
                parameters = @{
                    strict_required_status_checks_policy = $true
                    required_status_checks = @(
                        @{ context = "🔄 CI / 🏷️ Compute Version" },
                        @{ context = "🔄 CI / 🎨 Lint & Type Check" },
                        @{ context = "🔄 CI / 🧪 Unit Tests" }
                    )
                }
            }
        )
    } | ConvertTo-Json -Depth 8

    $payload | gh api "repos/$REPO/rulesets" --method POST --input - | Out-Null
    Write-Ok "Ruleset created for '$Pattern'"
}

# ─── main ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "============================================================================"
Write-Host "  🔒 cephnet — GitFlow Branch Protection Setup"
Write-Host "  Repo: https://github.com/$REPO"
Write-Host "============================================================================"
Write-Host ""

Test-Requirements

Write-Host ""
Write-Host "📌 Step 1: Ensure required branches exist"
Write-Host "─────────────────────────────────────────"
Ensure-BranchExists "main"
Ensure-BranchExists "dev"

Write-Host ""
Write-Host "🔒 Step 2: Apply protection to permanent branches"
Write-Host "─────────────────────────────────────────────────"
Protect-Branch "main"
Protect-Branch "dev"

Write-Host ""
Write-Host "🔒 Step 3: Apply rulesets for pattern-based branches"
Write-Host "────────────────────────────────────────────────────"
Protect-Pattern "release/**"
Protect-Pattern "hotfix/**"

Write-Host ""
Write-Host "============================================================================"
Write-Host "  ✅ Branch protection setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Protected branches:"
Write-Host "    • main       — requires PR + 1 approval + CI pass"
Write-Host "    • dev        — requires PR + 1 approval + CI pass"
Write-Host "    • release/** — requires PR + 1 approval + CI pass"
Write-Host "    • hotfix/**  — requires PR + 1 approval + CI pass"
Write-Host "============================================================================"
