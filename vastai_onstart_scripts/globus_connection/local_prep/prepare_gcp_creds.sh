#!/bin/bash
# 本地一次性运行：在完成 GCP setup 后，打包凭证供 VastAI 使用。
#
# 前置步骤（需人工，只做一次）：
#   1. 下载 GCP: wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
#   2. 解压: tar xzf globusconnectpersonal-latest.tgz && cd globusconnectpersonal-*
#   3. 运行 setup（会打开浏览器）: ./globusconnectpersonal -setup
#   4. 启动 GCP: ./globusconnectpersonal -start &
#   5. 等待连接成功
#   6. 运行本脚本: bash prepare_gcp_creds.sh
#
# 输出：
#   - globus_creds.b64: base64 编码的凭证包，填入 VastAI 环境变量 GLOBUS_CREDS_B64
#   - 打印 endpoint UUID，可填入 GLOBUS_SOURCE_ENDPOINT（可选，脚本可自动检测）

set -e

GLOBUS_HOME="${HOME}/.globusonline"
OUTPUT_B64="globus_creds.b64"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$GLOBUS_HOME" ]; then
    echo "错误: $GLOBUS_HOME 不存在。请先完成 GCP setup。"
    exit 1
fi

echo "打包 $GLOBUS_HOME ..."
cd "$(dirname "$GLOBUS_HOME")"
if base64 --help 2>/dev/null | grep -q "\-w"; then
    tar czf - "$(basename "$GLOBUS_HOME")" | base64 -w 0 > "$SCRIPT_DIR/$OUTPUT_B64"
else
    tar czf - "$(basename "$GLOBUS_HOME")" | base64 > "$SCRIPT_DIR/$OUTPUT_B64"
fi

echo ""
echo "=============================================="
echo "凭证已保存到: $SCRIPT_DIR/$OUTPUT_B64"
echo "=============================================="
echo ""
echo "将文件内容填入 VastAI 环境变量 GLOBUS_CREDS_B64"
echo "（若 VastAI 有长度限制，可拆分为 GLOBUS_CREDS_B64_1, _2, _3 等）"
echo ""

# 尝试获取 endpoint UUID
if command -v globus &>/dev/null; then
    EP_ID=$(globus endpoint local-id 2>/dev/null || true)
    if [ -n "$EP_ID" ]; then
        echo "GCP Endpoint UUID: $EP_ID"
        echo "可填入 GLOBUS_SOURCE_ENDPOINT（可选，脚本会自动从凭证检测）"
        echo ""
    fi
fi

python3 -c "
try:
    from globus_sdk import LocalGlobusConnectPersonal
    ep = LocalGlobusConnectPersonal()
    eid = getattr(ep, 'endpoint_id', None)
    if eid:
        print('GCP Endpoint UUID (from SDK):', eid)
        print('可填入 GLOBUS_SOURCE_ENDPOINT（可选）')
except Exception:
    pass
" 2>/dev/null || true
