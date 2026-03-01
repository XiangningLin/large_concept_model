# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from fairseq2.assets import AssetCard
from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import MetricRecorder
from fairseq2.optim import DynamicLossScaler
from fairseq2.optim.lr_scheduler import AbstractLRScheduler
from fairseq2.utils.profiler import Profiler, Stopwatch
from fairseq2.utils.rng import RngBag
from omegaconf import MISSING
from stopes.core import Requirements
from torch.nn import Module
from torch.optim import Optimizer

from lcm.datasets.configs import ParquetDatasetConfig
from lcm.datasets.dataloader import LCMDataLoader
from lcm.datasets.dataloading import ds_name
from lcm.models.abstract_lcm import AbstractLCModelConfig
from lcm.models.base_lcm.loader import load_base_lcm_model
from lcm.train.criterion import CriterionsFactory
from lcm.train.metrics import LCMMetricBag
from lcm.train.mse_lcm.criterion import ReconstructionCriterionConfig
from lcm.train.trainer import Trainer, TrainerBuilder, TrainingConfig
from lcm.utils.card_utils import create_model_card
from lcm.utils.hf_upload import download_checkpoint_folder_from_hf

logger = get_log_writer(__name__)


@dataclass
class LCMTrainingConfig(TrainingConfig):
    """Holds the configuration of an LCM training job."""

    training_data: List[ParquetDatasetConfig] = field(default_factory=list)
    """The datasets to train with."""  # TODO use dataset cards

    validation_data: List[ParquetDatasetConfig] = field(default_factory=list)
    """The datasets to validate on."""  # TODO use dataset cards

    model_config_or_name: Union[AbstractLCModelConfig, str, None] = None
    """The model configuration or name to train."""

    requirements: Requirements = field(
        default_factory=lambda: Requirements(
            nodes=1,
            tasks_per_node=8,
            gpus_per_node=8,
            cpus_per_task=8,
            mem_gb=256,
            timeout_min=3 * 24 * 60,
            constraint="volta32gb",
        )
    )
    """The scheduling requirements for this trainer"""

    criterion: ReconstructionCriterionConfig = MISSING
    """The MSE loss is the default base criterion used in either the `lcm` or `mse_lcm` trainers"""

    max_subword_length: int = 512
    """ Max subword length used to truncate seqs during sonar decoder backprop"""


class LCMTrainer(Trainer):
    config: LCMTrainingConfig
    model: Module
    training_data_loader: LCMDataLoader
    validation_data_loader: Optional[LCMDataLoader]
    gang: Gang
    optimizer: Optimizer
    loss_scaler: DynamicLossScaler
    lr_scheduler: AbstractLRScheduler
    rng_bag: RngBag
    step_nr: int
    train_metric_bag: LCMMetricBag
    valid_metric_bag: Mapping[str, LCMMetricBag]
    metric_recorders: List[MetricRecorder]
    profiler: Profiler
    stopwatch: Stopwatch

    def __init__(
        self,
        config: LCMTrainingConfig,
        model: Module,
        training_data_loader: LCMDataLoader,
        validation_data_loader: Optional[LCMDataLoader],
        gang: Gang,
        checkpoint_manager: FileCheckpointManager,
        rng_bag: RngBag,
        stopwatch: Stopwatch,
        card_metadata: Dict,
    ) -> None:
        super().__init__(
            config,
            model,
            training_data_loader,
            validation_data_loader,
            gang,
            checkpoint_manager,
            rng_bag,
            stopwatch,
            card_metadata=card_metadata,
        )

    def setup_criterion(self):
        return CriterionsFactory.build_criterion(
            name=self.config.criterion.name,
            config=self.config.criterion,
            model=self.model,
        )

    def setup_metric_bags(self):
        self.train_metric_bag = LCMMetricBag(
            self.gang,
            loss_summands=self.criterion.summands,
            reduction=self.criterion.reduction,
        )

        self.register_non_stateful(
            "valid_metric_bag",
            {
                ds_name(dataset): LCMMetricBag(
                    self.gang,
                    loss_summands=self.criterion.summands,
                    reduction=self.criterion.reduction,
                )
                for dataset in self.config.validation_data
            },
        )

    def create_model_card_for_last_checkpoint(
        self, is_final: bool = True, **card_kwargs
    ) -> Optional[AssetCard]:
        """Create a model card based on the last saved
        checkpoint and the model config."""

        current_step_number: Optional[int] = None
        if is_final:
            steps = self.checkpoint_manager.get_step_numbers()
            current_step_number = steps[-1] if len(steps) else None
        else:
            current_step_number = self.checkpoint_manager._get_checkpoint_step_nr()

        if current_step_number is None:
            logger.warning(
                "No checkpoint was saved, the final model card wil not be created"
            )
            return None

        cp_fn = (
            self.checkpoint_manager._checkpoint_dir
            / f"step_{current_step_number}"
            / "model.pt"  # type: ignore
        )

        card = create_model_card(
            checkpoint_path=cp_fn.absolute(),
            model_arch=self.card_metadata["model_arch"],
            model_config=self.card_metadata["model_config"],
            model_type=self.card_metadata["model_type"],
            **card_kwargs,
        )
        return card


class LCMTrainerBuilder(TrainerBuilder):
    config: LCMTrainingConfig

    def __init__(self, config: LCMTrainingConfig):
        super().__init__(config)

    def load_data(self):
        """Load training and validation data"""

        training_data_loader = LCMDataLoader(
            data_config=self.config.data_loading_config,
            datasets=self.config.training_data,
            max_subword_length=self.config.max_subword_length,
            dtype=self.dtype,
            gang=self.gang,
        )

        validation_data_loader = LCMDataLoader(
            data_config=self.config.validation_data_loading_config,
            datasets=self.config.validation_data,
            max_subword_length=self.config.max_subword_length,
            dtype=self.dtype,
            gang=self.gang,
        )

        return training_data_loader, validation_data_loader

    @property
    def model_loader(self):
        """A fairseq2 ModelLoader"""
        return load_base_lcm_model

    def build_trainer(self):
        """Build the trainer by loading data and
        setting up the model for training"""

        training_data_loader, validation_data_loader = self.load_data()

        checkpoint_manager = FileCheckpointManager(
            self.config.output_dir.joinpath("checkpoints"),
            self.gang,
        )

        # ── Resume only when resume_from is explicitly set ──
        resume_from = getattr(self.config, "resume_from", None)
        if resume_from is None:
            self.has_checkpoint = False
        elif resume_from == "hf":
            # HF: download when hf_repo_id and hf_checkpoint_filename are set
            self.has_checkpoint = checkpoint_manager.has_checkpoint()
            if not self.has_checkpoint and self.config.hf_repo_id and self.config.hf_checkpoint_filename:
                base_subfolder = self.config.hf_checkpoint_subfolder or "checkpoints"
                subfolder = (
                    f"{base_subfolder}/{self.config.hf_checkpoint_filename}"
                    if self.config.hf_checkpoint_filename
                    else base_subfolder
                )
                logger.info(
                    f"Resume from HF: Downloading from {self.config.hf_repo_id}/{subfolder}"
                )
                try:
                    import shutil
                    downloaded_dir = download_checkpoint_folder_from_hf(
                        repo_id=self.config.hf_repo_id,
                        subfolder=subfolder,
                        local_dir=self.config.output_dir / "hf_downloads",
                        token=self.config.hf_token,
                    )
                    local_ckpt_dir = self.config.output_dir / "checkpoints"
                    local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    # downloaded_dir is the step_XXXX folder; copy into checkpoints/step_XXXX/
                    dest_step_dir = local_ckpt_dir / downloaded_dir.name
                    dest_step_dir.mkdir(parents=True, exist_ok=True)
                    for item in downloaded_dir.iterdir():
                        dest = dest_step_dir / item.name
                        if item.is_dir():
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(item, dest)
                        else:
                            shutil.copy2(item, dest)
                    self.has_checkpoint = checkpoint_manager.has_checkpoint()
                    logger.info(f"HF checkpoint downloaded. has_checkpoint={self.has_checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to download checkpoint from HF Hub: {e}")
        else:
            # Local path: load from resume_from directory
            resume_path = Path(resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(f"resume_from path does not exist: {resume_from}")
            resume_checkpoint_manager = FileCheckpointManager(resume_path, self.gang)
            self.has_checkpoint = resume_checkpoint_manager.has_checkpoint()
            if not self.has_checkpoint:
                raise RuntimeError(
                    f"No checkpoint found in resume_from directory: {resume_from}"
                )
            # Store for use after trainer.setup()
            self._resume_checkpoint_manager = resume_checkpoint_manager

        model = self.create_model()

        model = self.maybe_load_model(model)

        model = self.maybe_freeze_parameters(model)

        # If using the META device, we need to move the model to gang.device
        wrapped_model = None

        if self.use_fsdp:
            wrapped_model = self.wrap_model_with_fsdp(model)
        elif self.use_ddp:
            wrapped_model = self.wrap_model_with_ddp(model)  # type: ignore

        trainer = LCMTrainer(
            self.config,  # type: ignore
            wrapped_model or model,
            training_data_loader,
            validation_data_loader,
            self.gang,
            checkpoint_manager,
            self.rng_bag,
            self.stopwatch,
            card_metadata=self.card_metadata,
        )

        trainer.setup()

        if self.has_checkpoint:
            load_manager = getattr(self, "_resume_checkpoint_manager", checkpoint_manager)
            resume_step = getattr(self.config, "resume_step", None)
            if self.config.training_mode == "decay_from_pretrain":
                self._load_pretrain_for_decay(trainer, load_manager, resume_step)
            elif load_manager is checkpoint_manager:
                trainer.restore(step_nr=resume_step)
            else:
                # Local resume_from: manually load (trainer.restore uses main checkpoint_manager)
                self._restore_from_manager(trainer, load_manager, resume_step)
            # Clear temp ref
            if hasattr(self, "_resume_checkpoint_manager"):
                delattr(self, "_resume_checkpoint_manager")

        return trainer

    def _load_pretrain_for_decay(
        self,
        trainer: LCMTrainer,
        checkpoint_manager: FileCheckpointManager,
        resume_step: Optional[int] = None,
    ) -> None:
        """Load a pretrain checkpoint for decay_from_pretrain mode.

        Loads model + optimizer weights from the pretrain checkpoint, keeps
        the freshly-built full-horizon WSD scheduler, and advances it to the
        checkpoint step so the LR curve picks up exactly where pretrain left
        off (stable phase continues if step < 0.9S, then decays to S).
        """
        logger.info("Loading pretrain checkpoint for decay_from_pretrain mode.")

        if resume_step is not None:
            if not checkpoint_manager.has_checkpoint(resume_step):
                raise RuntimeError(
                    f"Checkpoint for step {resume_step} not found in "
                    f"{checkpoint_manager._checkpoint_dir}"
                )
            step_nr = resume_step
            checkpoint = checkpoint_manager.load_checkpoint(resume_step)
        else:
            step_nr, checkpoint = checkpoint_manager.load_last_checkpoint()

        logger.info(f"Pretrain checkpoint loaded from step {step_nr}.")

        if "lr_scheduler" in checkpoint:
            checkpoint.pop("lr_scheduler")
            logger.info("Skipping pretrain LR scheduler state (using full-horizon decay schedule).")
        checkpoint["lr_scheduler"] = trainer.lr_scheduler.state_dict()

        trainer.load_state_dict(checkpoint)

        # Advance the fresh full-horizon WSD scheduler to the checkpoint step
        # so its internal counter matches the training position.
        for _ in range(step_nr):
            trainer.lr_scheduler.step()
        logger.info(
            f"LR scheduler advanced to step {step_nr} "
            f"(last_epoch={trainer.lr_scheduler.last_epoch})."
        )

        try:
            metadata = checkpoint_manager.load_metadata(step_nr)
            if metadata is not None and "total_tokens_seen" in metadata:
                trainer.total_tokens_seen = metadata["total_tokens_seen"]
                logger.info(
                    f"Inherited total_tokens_seen={trainer.total_tokens_seen:,} from pretrain."
                )
        except Exception:
            logger.warning("Could not restore milestone state from pretrain checkpoint.")

        trainer.saved_milestones = set()

        trainer.gang.barrier()

        decay_steps = int(trainer.config.max_steps * trainer.config.decay_ratio)
        trainer.step_nr = step_nr + 1

        logger.info(
            f"Decay-from-pretrain setup complete. "
            f"Resuming from step {trainer.step_nr}, "
            f"max_steps={trainer.config.max_steps} (S), "
            f"decay starts at step {trainer.config.pretrain_total_steps} (0.9S), "
            f"decay_steps={decay_steps}."
        )

    def _restore_from_manager(
        self,
        trainer: LCMTrainer,
        load_manager: FileCheckpointManager,
        resume_step: Optional[int] = None,
    ) -> None:
        """Restore trainer state from a checkpoint manager (e.g. resume_from local path)."""
        logger.info("Loading checkpoint from resume_from path.")
        if resume_step is not None:
            if not load_manager.has_checkpoint(resume_step):
                raise RuntimeError(
                    f"Checkpoint for step {resume_step} not found in "
                    f"{load_manager._checkpoint_dir}"
                )
            step_nr = resume_step
            checkpoint = load_manager.load_checkpoint(resume_step)
        else:
            step_nr, checkpoint = load_manager.load_last_checkpoint()
        logger.info(f"Checkpoint loaded, restoring training from step {step_nr}.")
        trainer.load_state_dict(checkpoint)
        trainer.gang.barrier()
        logger.info("Training restored, resuming.")
        trainer.step_nr = step_nr + 1
        try:
            metadata = load_manager.load_metadata(step_nr)
            if metadata is not None:
                if "total_tokens_seen" in metadata:
                    trainer.total_tokens_seen = metadata["total_tokens_seen"]
                    logger.info(f"Restored total_tokens_seen={trainer.total_tokens_seen:,}")
                if "saved_milestones" in metadata:
                    trainer.saved_milestones = set(metadata["saved_milestones"])
                    logger.info(f"Restored saved_milestones={trainer.saved_milestones}")
        except Exception:
            logger.warning("Could not restore milestone state from checkpoint metadata.")


def prepare_lcm_trainer(config: LCMTrainingConfig) -> LCMTrainer:
    """Create an LCM Trainer.
    :param config: The training configuration.
    """
    return LCMTrainerBuilder(config).build_trainer()
