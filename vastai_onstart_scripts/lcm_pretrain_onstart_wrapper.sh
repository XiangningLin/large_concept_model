#!/bin/bash
# LCM Pretrain OnStart Wrapper (VastAI CLI compatible - under 4048 chars)
# This lightweight wrapper downloads and executes the full setup script
#
# REQUIRED: Set GITHUB_TOKEN in VastAI Environment Variables (Web UI or Account Settings)
#   - Web UI: Instance Edit -> Environment Variables
#   - Account-level: vastai create env-var GITHUB_TOKEN <your_token>
#   - Per-instance CLI: -e GITHUB_TOKEN=xxx in docker options

set -e
exec 1> >(tee -a /workspace/lcm_startup.log)
exec 2>&1

echo "======================================"
echo "[$(date)] LCM Pretrain Startup Wrapper"
echo "======================================"

# GITHUB_TOKEN required if repo is private (XiangningLin/large_concept_model)
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
BRANCH_NAME="${BRANCH_NAME:-jianwen-modified}"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: GITHUB_TOKEN is not set. Set it in VastAI Environment Variables."
    echo "  Web UI: Instance Edit -> Environment Variables -> GITHUB_TOKEN=..."
    echo "  Or: vastai create env-var GITHUB_TOKEN <your_token> (account-level)"
    exit 1
fi

# Use GitHub API for private repo (raw.githubusercontent.com returns 404 for private repos)
# Note: Fine-grained PATs (github_pat_*) require "Authorization: Bearer", not "token"
SCRIPT_PATH="/tmp/lcm_pretrain_full.sh"

download_script() {
    local ref="$1"
    local url="https://api.github.com/repos/XiangningLin/large_concept_model/contents/vastai_onstart_scripts/lcm_pretrain_onstart.sh?ref=${ref}"
    echo "[$(date)] Trying branch: $ref"
    curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3.raw" \
        "$url" -o "$SCRIPT_PATH"
}

echo "[$(date)] Downloading full pretrain script via GitHub API..."
if ! download_script "$BRANCH_NAME"; then
    echo "[$(date)] $BRANCH_NAME failed, trying main..."
    if ! download_script "main"; then
        echo "ERROR: Failed on both $BRANCH_NAME and main. Check: 1) GITHUB_TOKEN valid + Contents:Read 2) File exists on remote"
        exit 1
    fi
fi

if [ ! -f "$SCRIPT_PATH" ] || [ ! -s "$SCRIPT_PATH" ]; then
    echo "ERROR: Downloaded file empty or missing"
    exit 1
fi

echo "[$(date)] Script downloaded successfully ($(wc -c < $SCRIPT_PATH) bytes)"
echo "[$(date)] Starting pretraining..."

# Execute the full script
bash "$SCRIPT_PATH"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "[$(date)] ✅ Pretraining completed successfully!"
    echo "======================================"
else
    echo "======================================"
    echo "[$(date)] ❌ Pretraining failed with exit code $EXIT_CODE"
    echo "======================================"
fi

exit $EXIT_CODE
