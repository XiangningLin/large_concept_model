#!/usr/bin/env python3
"""
本地诊断脚本：用 refresh_token 尝试获取 Delta endpoint 信息并模拟 transfer，
捕获 ConsentRequired 的完整错误（含 required_scopes），便于排查。

用法：
  export GLOBUS_REFRESH_TOKEN="你的token"
  export GLOBUS_CLIENT_ID="你的client_id"
  python diagnose_consent.py [dest_endpoint_uuid]

默认 dest_endpoint: 2d66a243-4a3f-4578-9d7f-1935fb5fba8f (NCSA Delta)
"""
from __future__ import annotations

import os
import sys

try:
    import globus_sdk
except ImportError:
    print("请先安装: pip install globus-sdk", file=sys.stderr)
    sys.exit(1)

DEST = os.environ.get("GLOBUS_DEST_ENDPOINT") or (sys.argv[1] if len(sys.argv) > 1 else "2d66a243-4a3f-4578-9d7f-1935fb5fba8f")
TOKEN = os.environ.get("GLOBUS_REFRESH_TOKEN")
CLIENT_ID = os.environ.get("GLOBUS_CLIENT_ID")

if not TOKEN or not CLIENT_ID:
    print("请设置 GLOBUS_REFRESH_TOKEN 和 GLOBUS_CLIENT_ID", file=sys.stderr)
    sys.exit(1)

auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
authorizer = globus_sdk.RefreshTokenAuthorizer(TOKEN, auth_client)
tc = globus_sdk.TransferClient(authorizer=authorizer)

print(f"目标 endpoint: {DEST}")
print("1. 获取 endpoint 信息...")
try:
    ep = tc.get_endpoint(DEST)
    print(f"   display_name: {ep.get('display_name')}")
    print(f"   entity_type: {ep.get('entity_type')}")
    print(f"   high_assurance: {ep.get('high_assurance')}")
    print(f"   uses_data_access: {ep.get('entity_type') == 'GCSv5_mapped_collection' and not ep.get('high_assurance')}")
except globus_sdk.TransferAPIError as e:
    print(f"   获取 endpoint 失败: {e}")
    if hasattr(e, "info") and e.info:
        cr = getattr(e.info, "consent_required", None)
        if cr and getattr(cr, "required_scopes", None):
            print(f"   required_scopes: {cr.required_scopes}")
    sys.exit(1)

print("\n2. 尝试 ls 目标路径（触发 consent 检查）...")
try:
    # 用根路径或用户路径触发
    tc.operation_ls(DEST, path="/")
    print("   ls 成功，consent 已满足")
except globus_sdk.TransferAPIError as e:
    print(f"   ls 失败: {e.code} - {e.message}")
    if hasattr(e, "info") and e.info:
        cr = getattr(e.info, "consent_required", None)
        if cr:
            print(f"\n   === ConsentRequired 详情 ===")
            print(f"   required_scopes: {getattr(cr, 'required_scopes', 'N/A')}")
        ap = getattr(e.info, "authorization_parameters", None)
        if ap:
            print(f"   authorization_parameters: {ap}")
    if hasattr(e, "raw_json") and e.raw_json:
        print(f"\n   完整 raw_json:")
        import json
        print(json.dumps(e.raw_json, indent=2))
    sys.exit(1)

print("\n诊断完成：当前 token 对目标 endpoint 的 consent 正常。")
