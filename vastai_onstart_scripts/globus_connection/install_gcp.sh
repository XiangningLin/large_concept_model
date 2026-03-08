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
# 显式指定 -d /home/globus，避免 VastAI 环境给新用户分配 /u/xxx 等错误 home
if ! id -u "$GCP_USER" &>/dev/null; then
    echo "[Globus] Creating user $GCP_USER..."
    useradd -m -d /home/globus -s /bin/bash "$GCP_USER" 2>/dev/null || {
        echo "[Globus] WARNING: Could not create user $GCP_USER (useradd failed). GCP may not start."
    }
fi
# 强制使用 /home/globus（VastAI 上 /u/jlyu3 可能不存在或不可写）
GCP_USER_HOME="/home/globus"
mkdir -p "$GCP_USER_HOME"
chown "$GCP_USER:$GCP_USER" "$GCP_USER_HOME" 2>/dev/null || true
if [ ! -d "$GCP_USER_HOME" ]; then
    echo "[Globus] ERROR: Cannot use $GCP_USER_HOME"
    exit 1
fi
echo "[Globus] Using GCP home: $GCP_USER_HOME"

# Restore ~/.globusonline to globus user's home (not root's)
echo "$CREDS" | base64 -d | tar xz -C "$GCP_USER_HOME"
if [ ! -d "$GCP_USER_HOME/.globusonline" ]; then
    echo "[Globus] ERROR: Failed to restore ~/.globusonline for $GCP_USER"
    exit 1
fi
chown -R "$GCP_USER:$GCP_USER" "$GCP_USER_HOME/.globusonline"

# 替换凭证包中的硬编码路径（prepare 时若 home 为 /u/jlyu3 等，配置会写入该路径）
# 可通过 GLOBUS_CREDS_OLD_HOME 自定义原路径，默认 /u/jlyu3
CREDS_OLD_HOME="${GLOBUS_CREDS_OLD_HOME:-/u/jlyu3}"
if [ "$CREDS_OLD_HOME" != "$GCP_USER_HOME" ] && grep -rq "$CREDS_OLD_HOME" "$GCP_USER_HOME/.globusonline" 2>/dev/null; then
    echo "[Globus] Patching config: $CREDS_OLD_HOME -> $GCP_USER_HOME"
    grep -rl "$CREDS_OLD_HOME" "$GCP_USER_HOME/.globusonline" 2>/dev/null | while read -r f; do
        sed -i "s|$CREDS_OLD_HOME|$GCP_USER_HOME|g" "$f" 2>/dev/null || true
    done
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
# 使用 -dir 显式指定配置目录（GCP 官方支持）
GCP_CONFIG_DIR="$GCP_USER_HOME/.globusonline"
runuser -u "$GCP_USER" -- "$GCP_BIN" -dir "$GCP_CONFIG_DIR" -stop 2>/dev/null || true
sleep 2
echo "[Globus] Starting GCP as user $GCP_USER (-dir=$GCP_CONFIG_DIR)..."
runuser -u "$GCP_USER" -- "$GCP_BIN" -dir "$GCP_CONFIG_DIR" -start &
sleep 5

# Verify it's running
if runuser -u "$GCP_USER" -- "$GCP_BIN" -dir "$GCP_CONFIG_DIR" -status 2>/dev/null | grep -q "connected"; then
    echo "[Globus] GCP installed and running"
else
    echo "[Globus] WARNING: GCP may have failed to start (check logs above)"
fi

echo "======================================"
