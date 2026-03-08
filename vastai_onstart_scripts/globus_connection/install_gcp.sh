#!/bin/bash
# Globus Connect Personal - restore pre-packaged credentials and start
# GCP refuses to run as root; we use a dedicated 'globus' user.
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
GCP_USER="globus"

echo "======================================"
echo "[Globus] Restoring GCP credentials and starting"
echo "======================================"

# Create non-root user (GCP refuses to run as root)
if ! id -u "$GCP_USER" &>/dev/null; then
    echo "[Globus] Creating user $GCP_USER..."
    useradd -m -s /bin/bash "$GCP_USER" 2>/dev/null || {
        echo "[Globus] WARNING: Could not create user $GCP_USER (useradd failed). GCP may not start."
    }
fi

GCP_USER_HOME=$(getent passwd "$GCP_USER" 2>/dev/null | cut -d: -f6)
if [ -z "$GCP_USER_HOME" ] || [ ! -d "$GCP_USER_HOME" ]; then
    echo "[Globus] ERROR: User $GCP_USER has no home directory"
    exit 1
fi

# Restore ~/.globusonline to globus user's home (not root's)
echo "$CREDS" | base64 -d | tar xz -C "$GCP_USER_HOME"
if [ ! -d "$GCP_USER_HOME/.globusonline" ]; then
    echo "[Globus] ERROR: Failed to restore ~/.globusonline for $GCP_USER"
    exit 1
fi
chown -R "$GCP_USER:$GCP_USER" "$GCP_USER_HOME/.globusonline"

# Download and extract GCP binary
mkdir -p "$GCP_INSTALL_DIR"
cd "$GCP_INSTALL_DIR"

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
chmod 755 "$GCP_BIN"
chown -R "$GCP_USER:$GCP_USER" "$GCP_INSTALL_DIR" 2>/dev/null || true

# Configure accessible paths in globus user's ~/.globusonline
CONFIG_PATHS="$GCP_USER_HOME/.globusonline/lta/config-paths"
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
chown -R "$GCP_USER:$GCP_USER" "$GCP_USER_HOME/.globusonline"

# Ensure globus can read the source path (create if needed, set perms)
mkdir -p "$ACCESSIBLE_DIR"
chmod 755 "$ACCESSIBLE_DIR" 2>/dev/null || true

# Stop any existing instance, then start as globus user
runuser -u "$GCP_USER" -- "$GCP_BIN" -stop 2>/dev/null || true
sleep 2
echo "[Globus] Starting GCP as user $GCP_USER..."
runuser -u "$GCP_USER" -- "$GCP_BIN" -start &
sleep 5

# Verify it's running
if runuser -u "$GCP_USER" -- "$GCP_BIN" -status 2>/dev/null | grep -q "connected"; then
    echo "[Globus] GCP installed and running"
else
    echo "[Globus] GCP started but may not be connected yet (check -status)"
fi

echo "======================================"
