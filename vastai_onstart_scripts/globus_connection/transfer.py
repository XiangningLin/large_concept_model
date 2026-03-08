#!/usr/bin/env python3
"""
Submit a Globus transfer from local GCP endpoint to a destination (e.g. NCSA Delta).
Exits 0 if transfer is skipped (not configured); exits 1 on error.
Uses NativeAppAuthClient by default; falls back to ConfidentialAppAuthClient if client_secret is set.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add parent so we can import config when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import GlobusConfig


def get_source_endpoint_id(cfg: GlobusConfig) -> str | None:
    """Get source endpoint UUID: from env, from file, or from LocalGlobusConnectPersonal."""
    if cfg.source_endpoint:
        return cfg.source_endpoint
    if os.path.isfile(cfg.endpoint_id_file):
        with open(cfg.endpoint_id_file) as f:
            return f.read().strip() or None
    try:
        from globus_sdk import LocalGlobusConnectPersonal
        local_ep = LocalGlobusConnectPersonal()
        return getattr(local_ep, "endpoint_id", None) if local_ep else None
    except Exception:
        return None


def main() -> int:
    cfg = GlobusConfig.from_env()

    if not cfg.is_transfer_configured():
        print("[Globus] Transfer not configured (need GLOBUS_REFRESH_TOKEN, GLOBUS_CLIENT_ID)")
        return 0

    try:
        import globus_sdk
    except ImportError:
        print("[Globus] globus-sdk not installed. Run: pip install globus-sdk", file=sys.stderr)
        return 1

    source_ep = get_source_endpoint_id(cfg)
    if not source_ep:
        print("[Globus] Could not determine source endpoint. Set GLOBUS_SOURCE_ENDPOINT or ensure GCP is running.", file=sys.stderr)
        return 1

    if not Path(cfg.source_path).exists():
        print(f"[Globus] Source path does not exist: {cfg.source_path}", file=sys.stderr)
        return 1

    print("======================================")
    print("[Globus] Submitting transfer to Delta")
    print("======================================")
    print(f"  Source: {source_ep}:{cfg.source_path}")
    print(f"  Dest:   {cfg.dest_endpoint}:{cfg.dest_path}")

    try:
        # Use ConfidentialAppAuthClient if client_secret is set; else NativeAppAuthClient
        if cfg.client_secret:
            auth_client = globus_sdk.ConfidentialAppAuthClient(
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
            )
        else:
            auth_client = globus_sdk.NativeAppAuthClient(cfg.client_id)

        authorizer = globus_sdk.RefreshTokenAuthorizer(
            cfg.refresh_token,
            auth_client,
        )
        tc = globus_sdk.TransferClient(authorizer=authorizer)

        # globus-sdk v4: TransferData(source, dest) 不再接受 tc 作为第一参数
        task_data = globus_sdk.TransferData(
            source_ep,
            cfg.dest_endpoint,
            label=cfg.transfer_label,
            sync_level=cfg.sync_level,
        )
        task_data.add_item(cfg.source_path, cfg.dest_path, recursive=True)

        result = tc.submit_transfer(task_data)
        task_id = result.get("task_id", "unknown")
        print(f"[Globus] Transfer submitted: {task_id}")
        print(f"  Monitor at: https://app.globus.org/activity/{task_id}")
        print("======================================")

        if cfg.wait_for_completion:
            print("[Globus] Waiting for transfer to complete...")
            while True:
                task = tc.get_task(task_id)
                status = task.get("status", "UNKNOWN")
                print(f"  Status: {status}")
                if status in ("SUCCEEDED", "FAILED"):
                    if status == "FAILED":
                        print("[Globus] Transfer failed. Check app.globus.org for details.", file=sys.stderr)
                        return 1
                    print("[Globus] Transfer completed successfully.")
                    break
                time.sleep(60)

        return 0

    except globus_sdk.TransferAPIError as e:
        if e.info and getattr(e.info, "consent_required", None):
            print("[Globus] ConsentRequired: You must grant consent for the destination endpoint.", file=sys.stderr)
            print("  Login at app.globus.org and consent to the endpoint, then retry.", file=sys.stderr)
        else:
            print(f"[Globus] TransferAPIError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[Globus] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
