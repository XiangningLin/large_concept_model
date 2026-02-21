# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

"""
HuggingFace Hub Checkpoint Utilities for LCM.

Provides upload and download functions for checkpoints on HuggingFace Hub.
All checkpoint persistence can go through HF Hub, removing
the need for local checkpoint storage between runs.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fairseq2.logging import get_log_writer

logger = get_log_writer(__name__)


def upload_checkpoint_to_hf(
    checkpoint_dir: Path,
    repo_id: str,
    path_in_repo: str,
    token: Optional[str] = None,
) -> None:
    """Upload a checkpoint directory to HuggingFace Hub.

    This function uploads the entire checkpoint folder (containing model weights,
    optimizer state, etc.) to the specified HF Hub repository.

    Args:
        checkpoint_dir: Local path to the checkpoint directory to upload.
        repo_id: HuggingFace repository ID (e.g., ``"myorg/my-lcm-model"``).
        path_in_repo: Destination path within the repository
            (e.g., ``"checkpoints/step_1000"``).
        token: HuggingFace API token. If ``None``, uses the ``HF_TOKEN``
            environment variable or cached credentials.

    Note:
        Upload failures are logged as warnings and do **not** interrupt training.
    """
    try:
        from huggingface_hub import HfApi

        token = token or os.environ.get("HF_TOKEN", None)
        api = HfApi(token=token)

        # Ensure the repo exists (create if needed)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(checkpoint_dir),
            path_in_repo=path_in_repo,
            repo_type="model",
            token=token,
        )

        logger.info(
            f"Checkpoint uploaded to HuggingFace Hub: {repo_id}/{path_in_repo}"
        )

    except Exception as e:
        logger.warning(
            f"Failed to upload checkpoint to HuggingFace Hub: {e}. "
            "Training will continue without upload."
        )


def download_checkpoint_from_hf(
    repo_id: str,
    subfolder: str,
    filename: str,
    local_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Path:
    """Download a checkpoint file from HuggingFace Hub.

    Downloads the specified checkpoint file to a local directory so it can be
    loaded by fairseq2's ``FileCheckpointManager``.

    Args:
        repo_id: HuggingFace repository ID (e.g., ``"myorg/my-lcm-model"``).
        subfolder: Subfolder within the repository
            (e.g., ``"checkpoints/ssm_370m_pretrain"``).
        filename: Name of the checkpoint file to download
            (e.g., ``"checkpoint_step_10000.pt"``).
        local_dir: Local directory to download to. If ``None``, uses a
            temporary directory.
        token: HuggingFace API token. If ``None``, uses the ``HF_TOKEN``
            environment variable or cached credentials.

    Returns:
        Path to the downloaded checkpoint file.

    Raises:
        RuntimeError: If the download fails.
    """
    try:
        from huggingface_hub import hf_hub_download

        token = token or os.environ.get("HF_TOKEN", None)

        if local_dir is None:
            local_dir = Path(tempfile.mkdtemp(prefix="lcm_hf_ckpt_"))

        path_in_repo = f"{subfolder}/{filename}" if subfolder else filename

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            local_dir=str(local_dir),
            token=token,
            repo_type="model",
        )

        logger.info(f"Checkpoint downloaded from HuggingFace Hub: {local_path}")
        return Path(local_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to download checkpoint from HuggingFace Hub "
            f"(repo={repo_id}, file={subfolder}/{filename}): {e}"
        ) from e


def download_checkpoint_folder_from_hf(
    repo_id: str,
    subfolder: str,
    local_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Path:
    """Download an entire checkpoint folder from HuggingFace Hub.

    This is useful for downloading fairseq2-style checkpoint directories
    (containing ``state.pt``, ``model.pt``, ``metadata.pt``, etc.).

    Args:
        repo_id: HuggingFace repository ID.
        subfolder: Subfolder within the repository to download
            (e.g., ``"checkpoints/step_10000"``).
        local_dir: Local directory to download to. If ``None``, uses a
            temporary directory.
        token: HuggingFace API token.

    Returns:
        Path to the downloaded checkpoint directory.
    """
    try:
        from huggingface_hub import snapshot_download

        token = token or os.environ.get("HF_TOKEN", None)

        if local_dir is None:
            local_dir = Path(tempfile.mkdtemp(prefix="lcm_hf_ckpt_"))

        local_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{subfolder}/**",
            local_dir=str(local_dir),
            token=token,
            repo_type="model",
        )

        result_dir = Path(local_path) / subfolder
        logger.info(
            f"Checkpoint folder downloaded from HuggingFace Hub: {result_dir}"
        )
        return result_dir

    except Exception as e:
        raise RuntimeError(
            f"Failed to download checkpoint folder from HuggingFace Hub "
            f"(repo={repo_id}, subfolder={subfolder}): {e}"
        ) from e
