#!/usr/bin/env bash
# setup-branch-protection.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-19
# Last modified: 2026-04-19
# Description: Configures GitHub branch protection rules for GitFlow branches.
#              Requires 'gh' CLI authenticated with repo admin permissions.
#              Usage: ./scripts/linux/setup-branch-protection.sh

set -euo pipefail

readonly REPO="pvagnozzi/cephnet"
readonly SCRIPT_NAME="$(basename "$0")"

# ─── helpers ──────────────────────────────────────────────────────────────────

log()  { echo "  $*"; }
ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠️  $*" >&2; }
fail() { echo "  ❌ $*" >&2; exit 1; }

check_requirements() {
    command -v gh   &>/dev/null || fail "'gh' CLI is not installed. See https://cli.github.com"
    command -v jq   &>/dev/null || fail "'jq' is not installed."
    gh auth status  &>/dev/null || fail "Not authenticated. Run: gh auth login"
    log "Requirements OK"
}

ensure_branch_exists() {
    local branch="$1"
    if gh api "repos/${REPO}/branches/${branch}" &>/dev/null; then
        ok "Branch '${branch}' already exists"
    else
        warn "Branch '${branch}' does not exist — creating from default branch"
        local default_branch
        default_branch="$(gh api "repos/${REPO}" --jq '.default_branch')"
        local sha
        sha="$(gh api "repos/${REPO}/git/ref/heads/${default_branch}" --jq '.object.sha')"
        gh api "repos/${REPO}/git/refs" \
            --method POST \
            --field "ref=refs/heads/${branch}" \
            --field "sha=${sha}" \
            > /dev/null
        ok "Branch '${branch}' created from '${default_branch}'"
    fi
}

protect_branch() {
    local branch="$1"
    local strict="${2:-true}"    # require branches to be up to date
    local dismissals="${3:-false}"

    log "Configuring protection for '${branch}'..."

    gh api "repos/${REPO}/branches/${branch}/protection" \
        --method PUT \
        --field "required_status_checks[strict]=${strict}" \
        --field "required_status_checks[contexts][]=🔄 CI / 🏷️ Compute Version" \
        --field "required_status_checks[contexts][]=🔄 CI / 🎨 Lint & Type Check" \
        --field "required_status_checks[contexts][]=🔄 CI / 🧪 Unit Tests" \
        --field "enforce_admins=false" \
        --field "required_pull_request_reviews[required_approving_review_count]=1" \
        --field "required_pull_request_reviews[dismiss_stale_reviews]=true" \
        --field "required_pull_request_reviews[require_code_owner_reviews=false" \
        --field "required_linear_history=false" \
        --field "allow_force_pushes=false" \
        --field "allow_deletions=false" \
        --field "block_creations=false" \
        --field "required_conversation_resolution=true" \
        > /dev/null

    ok "Branch '${branch}' protected (PR required, status checks required)"
}

protect_pattern() {
    local pattern="$1"
    log "Configuring ruleset for pattern '${pattern}'..."

    # Use repository rulesets (modern GitHub branch protection)
    gh api "repos/${REPO}/rulesets" \
        --method POST \
        --input - <<JSON
{
  "name": "Protect ${pattern}",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["refs/heads/${pattern}"],
      "exclude": []
    }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true
      }
    },
    { "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          { "context": "🔄 CI / 🏷️ Compute Version" },
          { "context": "🔄 CI / 🎨 Lint & Type Check" },
          { "context": "🔄 CI / 🧪 Unit Tests" }
        ]
      }
    }
  ]
}
JSON
    ok "Ruleset created for '${pattern}'"
}

# ─── main ─────────────────────────────────────────────────────────────────────

main() {
    echo "============================================================================"
    echo "  🔒 cephnet — GitFlow Branch Protection Setup"
    echo "  Repo: https://github.com/${REPO}"
    echo "============================================================================"
    echo ""

    check_requirements

    echo ""
    echo "📌 Step 1: Ensure required branches exist"
    echo "─────────────────────────────────────────"
    ensure_branch_exists "main"
    ensure_branch_exists "dev"

    echo ""
    echo "🔒 Step 2: Apply protection to permanent branches"
    echo "─────────────────────────────────────────────────"
    protect_branch "main" "true"
    protect_branch "dev"  "true"

    echo ""
    echo "🔒 Step 3: Apply rulesets for pattern-based branches"
    echo "────────────────────────────────────────────────────"
    protect_pattern "release/**"
    protect_pattern "hotfix/**"

    echo ""
    echo "============================================================================"
    echo "  ✅ Branch protection setup complete!"
    echo ""
    echo "  Protected branches:"
    echo "    • main       — requires PR + 1 approval + CI pass"
    echo "    • dev        — requires PR + 1 approval + CI pass"
    echo "    • release/** — requires PR + 1 approval + CI pass"
    echo "    • hotfix/**  — requires PR + 1 approval + CI pass"
    echo "============================================================================"
}

main "$@"
