"""
Globus connection module - configuration from environment variables.
All config is optional; missing values cause the module to skip operations.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _get_creds_b64() -> str:
    """Read GLOBUS_CREDS_B64 (single var; creds in script for VastAI 16-env limit)."""
    return os.environ.get("GLOBUS_CREDS_B64") or ""


@dataclass
class GlobusConfig:
    """Globus configuration loaded from environment."""

    # GCP install (install_gcp.sh) - pre-packaged credentials
    creds_b64: Optional[str] = None
    install_dir: str = "/workspace/globus_gcp"
    source_path: str = "/workspace/lcm/preprocessed_data"

    # Transfer auth (transfer.py)
    refresh_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None  # Optional; use Confidential App if Native fails

    # Transfer target
    source_endpoint: Optional[str] = None
    dest_endpoint: str = "2d66a243-4a3f-4578-9d7f-1935fb5fba8f"  # NCSA Delta（2025 年有效）
    dest_path: str = "/work/hdd/bfaq/jlyu3/lcm/preprocessed_data"
    transfer_label: str = "VastAI-to-Delta"
    wait_for_completion: bool = False
    sync_level: str = "checksum"

    # Temp file for endpoint ID (optional)
    endpoint_id_file: str = "/tmp/globus_gcp_endpoint_id"

    @classmethod
    def from_env(cls) -> "GlobusConfig":
        return cls(
            creds_b64=_get_creds_b64() or None,
            install_dir=os.environ.get("GLOBUS_GCP_INSTALL_DIR", "/workspace/globus_gcp"),
            source_path=os.environ.get("GLOBUS_SOURCE_PATH", "/workspace/lcm/preprocessed_data"),
            refresh_token=os.environ.get("GLOBUS_REFRESH_TOKEN") or None,
            client_id=os.environ.get("GLOBUS_CLIENT_ID") or None,
            client_secret=os.environ.get("GLOBUS_CLIENT_SECRET") or None,
            source_endpoint=os.environ.get("GLOBUS_SOURCE_ENDPOINT") or None,
            dest_endpoint=os.environ.get("GLOBUS_DEST_ENDPOINT", "2d66a243-4a3f-4578-9d7f-1935fb5fba8f"),
            dest_path=os.environ.get("GLOBUS_DEST_PATH", "/work/hdd/bfaq/jlyu3/lcm/preprocessed_data"),
            transfer_label=os.environ.get("GLOBUS_TRANSFER_LABEL", "VastAI-to-Delta"),
            wait_for_completion=os.environ.get("GLOBUS_WAIT_FOR_COMPLETION", "").lower() in ("1", "true", "yes"),
            sync_level=os.environ.get("GLOBUS_SYNC_LEVEL", "checksum"),
            endpoint_id_file=os.environ.get("GLOBUS_ENDPOINT_ID_FILE", "/tmp/globus_gcp_endpoint_id"),
        )

    def is_install_configured(self) -> bool:
        """True if GCP install should run (has pre-packaged creds)."""
        return bool(self.creds_b64)

    def is_transfer_configured(self) -> bool:
        """True if transfer should run (has auth). Native App needs only refresh_token + client_id."""
        return bool(self.refresh_token and self.client_id)
