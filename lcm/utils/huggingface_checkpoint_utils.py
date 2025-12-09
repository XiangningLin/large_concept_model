"""
HuggingFace Hub Checkpoint Utilities

Simple utilities for saving checkpoints to HuggingFace Hub.

Note: Set HF_TOKEN environment variable or pass token directly.
"""

import torch
import os
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi
import tempfile
import shutil


class HuggingFaceCheckpointManager:
    """
    Manages checkpoint saving to HuggingFace Hub.
    
    Saves checkpoint to temporary file and uploads to Hub.
    """
    
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None
    ):
        """
        Args:
            repo_id: HuggingFace repo ID (e.g., "LGVamper/Sentence-SSM")
            token: HuggingFace token (None to use HF_TOKEN env var or cached token)
        """
        self.repo_id = repo_id
        # Priority: 1. Passed token, 2. HF_TOKEN env var, 3. Cached token (None)
        self.token = token or os.environ.get('HF_TOKEN', None)
        
        # Initialize HuggingFace API
        self.api = HfApi(token=self.token)
        
    
    def save_checkpoint(
        self,
        checkpoint: dict,
        step: int,
        loss: float,
        subfolder: str = "checkpoints"
    ) -> str:
        """
        Save checkpoint to HuggingFace Hub.
        
        Args:
            checkpoint: Checkpoint dictionary to save
            step: Training step
            loss: Current loss
            subfolder: Subfolder in the repo for checkpoints
            
        Returns:
            URL to the uploaded checkpoint
        """
        # Generate checkpoint filename
        filename = f"checkpoint_step_{step}_loss_{loss:.4f}.pth"
        
        # Use temporary directory
        temp_dir = tempfile.mkdtemp()
        local_path = Path(temp_dir) / filename
        
        try:
            # Save checkpoint to temporary file
            torch.save(checkpoint, local_path)
            
            # Upload to HuggingFace Hub
            path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
            
            url = self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token
            )
            
            print(f"Checkpoint uploaded to HuggingFace Hub: {url}")
            
            return url
            
        finally:
            # Cleanup temporary directory
            if local_path.exists() and local_path.parent.exists():
                shutil.rmtree(local_path.parent)

    def download_checkpoint(
        self,
        filename: str,
        subfolder: str = "checkpoints",
        map_location: str = "cpu"
    ) -> str:
        """
        Load checkpoint from HuggingFace Hub.
        
        Args:
            filename: Checkpoint filename
            subfolder: Subfolder in the repo
            map_location: Device to load checkpoint to
            
        Returns:
            Local path to downloaded checkpoint
        """
        from huggingface_hub import hf_hub_download
        
        # Download checkpoint
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=f"{subfolder}/{filename}" if subfolder else filename,
            token=self.token,
            repo_type="model"
        )
        
        return local_path


def save_checkpoint_to_hub(
    trainer,
    step: int,
    loss: float,
    repo_id: str = "LGVamper/Sentence-SSM",
    token: Optional[str] = None,
    checkpoint_subfolder: Optional[str] = None
) -> str:
    """
    Save checkpoint to HuggingFace Hub (Hub only, no local save).
    
    Args:
        trainer: MSETrainer instance
        step: Current training step
        loss: Current loss
        repo_id: HuggingFace repo ID
        token: HuggingFace token (None to use HF_TOKEN env var or config file)
        checkpoint_subfolder: Optional subfolder within checkpoints/ directory 
                              (e.g., "experiment1" will save to "checkpoints/experiment1/")
        
    Returns:
        URL to uploaded checkpoint
    """
    # Only main process uploads
    if trainer.accelerator and not trainer.accelerator.is_main_process:
        return None
    # Import here to avoid circular dependency
    from src.utils.training_utils import get_random_states
    from datetime import datetime
    
    # Collect checkpoint data (same structure as training_utils.save_checkpoint)
    rng_states = get_random_states()
    
    grad_scaler_state = None
    if trainer.accelerator and hasattr(trainer.accelerator, 'scaler') and trainer.accelerator.scaler is not None:
        try:
            grad_scaler_state = trainer.accelerator.scaler.state_dict()
        except Exception as e:
            # Silent failure, consistent with training_utils
            pass
    
    training_state = trainer.step_accounting.get_state_dict()
    
    checkpoint = {
        # Core training state
        'step': step,
        'epoch': training_state['current_epoch'],
        'loss': loss,
        'grad_accum_step': training_state['grad_accum_step'],
        
        # Model and optimizer
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict() if trainer.optimizer is not None else None,
        
        # Learning rate scheduler
        'scheduler_state_dict': trainer.lr_scheduler.state_dict() if trainer.lr_scheduler is not None else None,
        
        # Mixed precision
        'grad_scaler_state_dict': grad_scaler_state,
        
        # Training metrics
        'best_loss': training_state['best_loss'],
        'current_loss': training_state['current_loss'],
        
        # Random number generator states
        'rng_states': rng_states,
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'max_steps': trainer.max_steps,
    }
    
    # Build subfolder path
    if checkpoint_subfolder:
        subfolder = f"checkpoints/{checkpoint_subfolder}"
    else:
        subfolder = "checkpoints"
    
    # Upload to HuggingFace Hub
    manager = HuggingFaceCheckpointManager(
        repo_id=repo_id,
        token=token
    )
    
    url = manager.save_checkpoint(checkpoint, step, loss, subfolder=subfolder)
    
    return url


def load_checkpoint_from_hub(
    trainer=None,
    filename: Optional[str] = None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    checkpoint_subfolder: Optional[str] = None
) -> str:
    """
    Download checkpoint from HuggingFace Hub and return local path.
    
    This function only downloads the checkpoint file. The caller should use:
    - resume_from_checkpoint() for resuming training (loads all states)
    - load_model() for evaluation (loads only model weights)
    
    Args:
        trainer: Optional trainer instance (used to read hf_repo_id, hf_token, hf_checkpoint_subfolder)
        filename: Checkpoint filename (e.g., "checkpoint_step_10000_loss_0.0000.pth")
        repo_id: HuggingFace repo ID (None to use trainer's hf_repo_id or must be provided)
        token: HuggingFace token (None to use trainer's hf_token or HF_TOKEN env var)
        checkpoint_subfolder: Optional subfolder within checkpoints/ directory 
                             (e.g., "experiment1" will load from "checkpoints/experiment1/")
                             (None to use trainer's hf_checkpoint_subfolder)
    
    Returns:
        str: Local path to downloaded checkpoint file
    """
    # Use trainer's config if available, otherwise use provided parameters
    if trainer is not None:
        if repo_id is None and hasattr(trainer, 'hf_repo_id'):
            repo_id = trainer.hf_repo_id
        if token is None and hasattr(trainer, 'hf_token'):
            token = trainer.hf_token
        if checkpoint_subfolder is None and hasattr(trainer, 'hf_checkpoint_subfolder'):
            checkpoint_subfolder = trainer.hf_checkpoint_subfolder
    
    # If no repo_id or filename provided, raise error
    if repo_id is None or filename is None:
        raise ValueError("Both repo_id and filename must be provided (or set in trainer attributes)")
    
    # Build subfolder path
    if checkpoint_subfolder:
        subfolder = f"checkpoints/{checkpoint_subfolder}"
    else:
        subfolder = "checkpoints"
    
    # Download checkpoint from HuggingFace Hub
    manager = HuggingFaceCheckpointManager(
        repo_id=repo_id,
        token=token
    )
    
    map_location = 'cpu'  # Download to CPU, caller can specify device when loading
    local_path = manager.download_checkpoint(
        filename=filename,
        subfolder=subfolder,
        map_location=map_location
    )
    
    print(f"Downloaded checkpoint from HuggingFace Hub: {local_path}")
    
    return local_path
