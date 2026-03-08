#!/bin/bash
# Globus Connect Personal - restore pre-packaged credentials and start
# Skips silently if GLOBUS_CREDS_B64 is not set (VastAI 16 env limit: creds live in script).
# Usage: bash install_gcp.sh

set -e

CREDS="${GLOBUS_CREDS_B64:-}"
if [ -z "$CREDS" ]; then
    echo "[Globus] GLOBUS_CREDS_B64 not set, skipping GCP install"
    exit 0
fi

GCP_INSTALL_DIR="${GLOBUS_GCP_INSTALL_DIR:-/workspace/globus_gcp}"
SOURCE_PATH="${GLOBUS_SOURCE_PATH:-/workspace/lcm/preprocessed_data}"
ACCESSIBLE_DIR="$(dirname "$SOURCE_PATH")"

echo "======================================"
echo "[Globus] Restoring GCP credentials and starting"
echo "======================================"

# Restore ~/.globusonline from pre-packaged tarball
echo "$CREDS" | base64 -d | tar xz -C "$HOME"
if [ ! -d "$HOME/.globusonline" ]; then
    echo "[Globus] ERROR: Failed to restore ~/.globusonline"
    exit 1
fi

mkdir -p "$GCP_INSTALL_DIR"
cd "$GCP_INSTALL_DIR"

# Download and extract GCP binary if needed
GCP_DIR=""
for d in globusconnectpersonal-*; do
    if [ -d "$d" ] && [ -f "$d/globusconnectpersonal" ]; then
        GCP_DIR="$d"
        break
    fi
done

if [ -z "$GCP_DIR" ]; then
    echo "[Globus] Downloading GCP..."
    wget -q https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz -O gcp.tgz
    tar xzf gcp.tgz
    for d in globusconnectpersonal-*; do
        if [ -d "$d" ] && [ -f "$d/globusconnectpersonal" ]; then
            GCP_DIR="$d"
            break
        fi
    done
fi

if [ -z "$GCP_DIR" ] || [ ! -f "$GCP_DIR/globusconnectpersonal" ]; then
    echo "[Globus] ERROR: Could not find globusconnectpersonal after extract"
    exit 1
fi

GCP_BIN="$GCP_INSTALL_DIR/$GCP_DIR/globusconnectpersonal"
cd "$GCP_INSTALL_DIR/$GCP_DIR"

# Configure accessible paths - GCP only allows paths listed in config-paths
CONFIG_PATHS="$HOME/.globusonline/lta/config-paths"
mkdir -p "$(dirname "$CONFIG_PATHS")"
if [ -f "$CONFIG_PATHS" ]; then
    if ! grep -q "^${ACCESSIBLE_DIR}," "$CONFIG_PATHS" 2>/dev/null; then
        echo "${ACCESSIBLE_DIR},0,1" >> "$CONFIG_PATHS"
        echo "[Globus] Added $ACCESSIBLE_DIR to config-paths"
    fi
else
    echo "~/,0,1" > "$CONFIG_PATHS"
    echo "${ACCESSIBLE_DIR},0,1" >> "$CONFIG_PATHS"
    echo "[Globus] Created config-paths with $ACCESSIBLE_DIR"
fi

# Stop any existing instance, then start
"$GCP_BIN" -stop 2>/dev/null || true
sleep 2
echo "[Globus] Starting GCP..."
"$GCP_BIN" -start &
sleep 5

# Verify it's running
if "$GCP_BIN" -status 2>/dev/null | grep -q "connected"; then
    echo "[Globus] GCP installed and running"
else
    echo "[Globus] GCP started but may not be connected yet (check -status)"
fi

echo "======================================"
