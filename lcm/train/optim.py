# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

import math
from typing import Sequence, Tuple, Union

from fairseq2.logging import get_log_writer
from fairseq2.optim.lr_scheduler import (
    AbstractLRScheduler,
    CosineAnnealingLR,
    MyleLR,
    NoopLR,
    PolynomialDecayLR,
)
from torch.optim import Optimizer

logger = get_log_writer(__name__)


def _get_per_param_group(
    optimizer: Optimizer, name: str, value: Union[float, Sequence[float]]
) -> Sequence[float]:
    """Map a scalar or per-group value to a list of length num_param_groups."""
    num_param_groups = len(optimizer.param_groups)
    if isinstance(value, float):
        return [value] * num_param_groups
    if len(value) != num_param_groups:
        raise ValueError(
            f"The length of `{name}` must be equal to the number of parameter groups "
            f"({num_param_groups}), but is {len(value)} instead."
        )
    return value


class WSDLR(AbstractLRScheduler):
    """Warmup-Stable-Decay (WSD) learning rate scheduler.

    Same as fairseq2 TriStageLR but correctly handles decay_ratio=0: when the
    decay stage has zero steps, keeps the peak LR instead of dropping to final_lr.
    This allows pretrain to use warmup+stable only (e.g. [0.1, 0.9, 0.0]).
    """

    _num_steps: int
    _start_lr_scales: Sequence[float]
    _final_lr_scales: Sequence[float]
    _start_lrs: list
    _final_lrs: list
    _num_stage1_steps: int
    _num_stage2_steps: int
    _num_stage3_steps: int

    def __init__(
        self,
        optimizer: Optimizer,
        num_steps: int,
        stage_ratio: Tuple[float, float, float],
        *,
        start_lr_scale: Union[float, Sequence[float]] = 0.01,
        final_lr_scale: Union[float, Sequence[float]] = 0.01,
        last_epoch: int = -1,
    ) -> None:
        if not math.isclose((s := sum(stage_ratio)), 1.0):
            raise ValueError(
                f"The sum of `stage_ratio` values must be 1.0, but is {s} instead."
            )
        self._num_steps = num_steps
        self._start_lr_scales = _get_per_param_group(
            optimizer, "start_lr", start_lr_scale
        )
        self._final_lr_scales = _get_per_param_group(
            optimizer, "final_lr", final_lr_scale
        )
        self._start_lrs = []
        self._final_lrs = []
        self._num_stage1_steps = int(stage_ratio[0] * num_steps)
        self._num_stage2_steps = int(stage_ratio[1] * num_steps)
        self._num_stage3_steps = int(stage_ratio[2] * num_steps)
        super().__init__(optimizer, last_epoch)

    def _compute_lrs(self) -> list:
        base_lrs = self.base_lrs
        if not self._start_lrs:
            self._start_lrs = [
                s * b for s, b in zip(self._start_lr_scales, base_lrs)
            ]
        if not self._final_lrs:
            self._final_lrs = [
                s * b for s, b in zip(self._final_lr_scales, base_lrs)
            ]
        num_steps = self.last_epoch

        # Warmup stage
        if self._num_stage1_steps > 0 and num_steps < self._num_stage1_steps:
            c = num_steps / self._num_stage1_steps
            return [
                s + (b - s) * c
                for b, s in zip(base_lrs, self._start_lrs)
            ]
        num_steps -= self._num_stage1_steps

        # Stable stage
        if self._num_stage2_steps > 0 and num_steps < self._num_stage2_steps:
            return list(base_lrs)
        num_steps -= self._num_stage2_steps

        # Decay stage
        if self._num_stage3_steps == 0:
            return list(base_lrs)
        if num_steps < self._num_stage3_steps:
            c = num_steps / self._num_stage3_steps
            return [
                b * math.exp(math.log(f) * c)
                for b, f in zip(base_lrs, self._final_lr_scales)
            ]
        return list(self._final_lrs)


def build_lr_scheduler(
    optimizer: Optimizer,
    lr: float,
    warmup_steps: int,
    start_lr: float = 1e-7,
    final_lr: float = 1e-5,
    max_steps: int = 10_000,
    stage_ratio: Tuple[float, ...] = (0.1, 0.4, 0.5),
    schedule: str = "myle",
) -> AbstractLRScheduler:
    assert schedule in [
        "noop",
        "myle",
        "cosine",
        "wsd",
        "polynomial",
    ], (
        f"Cannot recognize the learing rate schedule {schedule}, only noop, myle, cosine and wsd are supported"
    )

    assert lr > 0, "The learning reate should be strictly positive"

    lr_scheduler: AbstractLRScheduler

    if schedule == "noop":
        lr_scheduler = NoopLR(optimizer)

    elif schedule == "myle":
        lr_scheduler = MyleLR(
            optimizer,
            num_warmup_steps=warmup_steps,
            start_lr=[start_lr],
        )

    elif schedule == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            cycle_len=max_steps - warmup_steps + 1,
            num_warmup_steps=warmup_steps,
            start_lr=[start_lr],
            final_lr=[final_lr],
            cycle_mul=1.0,
            lr_mul=1.0,
        )

    elif schedule == "wsd":
        assert lr > start_lr, (
            f"the starting learning rate {start_lr} should be lesser than the main lr {lr}"
        )
        start_lr_scale = start_lr / lr

        assert lr > final_lr, (
            f"the final learning rate {final_lr} should be lesser than the main lr {lr}"
        )
        final_lr_scale = final_lr / lr

        lr_scheduler = WSDLR(
            optimizer,
            max_steps,
            stage_ratio=tuple(stage_ratio),
            start_lr_scale=start_lr_scale,
            final_lr_scale=final_lr_scale,
        )

    elif schedule == "polynomial":
        lr_scheduler = PolynomialDecayLR(
            optimizer,
            max_steps,
            warmup_steps,
            power=200,
            start_lr=start_lr,
            final_lr=final_lr,
        )

    return lr_scheduler
