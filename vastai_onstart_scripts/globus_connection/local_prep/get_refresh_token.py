#!/usr/bin/env python3
"""
本地一次性运行：获取 Globus refresh token。
需要：pip install globus-sdk

用法：
  1. 去 developers.globus.org 创建 Native App（Redirect URI: https://auth.globus.org/v2/web/auth-code）
  2. 记下 client_id
  3. 运行: python get_refresh_token.py <client_id>
  4. 浏览器打开打印的 URL，登录并授权
  5. 将授权后页面显示的 code 粘贴回终端
  6. 复制输出的 refresh_token，存入 VastAI 环境变量 GLOBUS_REFRESH_TOKEN

注意：若后续 transfer 报错需 client_secret，则改用 Confidential App，并同时保存 client_secret。
"""
from __future__ import annotations

import sys

try:
    import globus_sdk
except ImportError:
    print("请先安装: pip install globus-sdk", file=sys.stderr)
    sys.exit(1)

# 从环境变量或命令行参数读取，否则提示输入
CLIENT_ID = None
if len(sys.argv) > 1:
    CLIENT_ID = sys.argv[1]
if not CLIENT_ID:
    import os
    CLIENT_ID = os.environ.get("GLOBUS_CLIENT_ID")

if not CLIENT_ID:
    print("用法: python get_refresh_token.py <client_id>")
    print("  或设置环境变量 GLOBUS_CLIENT_ID")
    print("  client_id 来自 developers.globus.org 创建的 Native App")
    sys.exit(1)

# NCSA Delta 等 mapped collection 需要 data_access scope（dependent scope 格式），否则 transfer 会报 ConsentRequired
# API 要求格式: urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/EP/data_access]
# 可通过环境变量 GLOBUS_DEST_ENDPOINT 覆盖
import os
DEST_ENDPOINT = os.environ.get("GLOBUS_DEST_ENDPOINT", "2d66a243-4a3f-4578-9d7f-1935fb5fba8f")

# 使用 dependent scope 格式（TransferScopes.all + data_access 依赖），否则 consent 不生效
from globus_sdk.scopes import GCSCollectionScopes, TransferScopes
data_access_scope = GCSCollectionScopes(DEST_ENDPOINT).data_access
transfer_scope = TransferScopes.all.with_dependency(data_access_scope.with_optional(True))

client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
client.oauth2_start_flow(
    requested_scopes=[transfer_scope],
    refresh_tokens=True,
)

print("\n请在浏览器打开以下 URL 并登录授权：")
print(client.oauth2_get_authorize_url())
print()
auth_code = input("授权后，将页面显示的 code 粘贴到这里: ").strip()

tokens = client.oauth2_exchange_code_for_tokens(auth_code)
transfer_tokens = tokens.by_resource_server.get("transfer.api.globus.org")
if not transfer_tokens:
    print("未获取到 transfer scope 的 token", file=sys.stderr)
    sys.exit(1)

refresh_token = transfer_tokens.get("refresh_token")
if not refresh_token:
    print("未获取到 refresh_token，请确保 oauth2_start_flow 中 refresh_tokens=True", file=sys.stderr)
    sys.exit(1)

print("\n" + "=" * 60)
print("你的 GLOBUS_REFRESH_TOKEN（请妥善保存，填入 VastAI 环境变量）：")
print("=" * 60)
print(refresh_token)
print("=" * 60)
