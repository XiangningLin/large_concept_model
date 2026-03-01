#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Multi-GPU Parallel FineWeb Data Preparation (stopes framework)

Uses stopes PartitionedDataMapper to automatically shard the HuggingFace
dataset across multiple GPUs.  Each shard runs FineWebPipeline which
performs: Unicode normalization -> sentence splitting -> truncation ->
UNK filtering -> SONAR encoding -> train/val split assignment.

Single-GPU fallback:  num_shards=1 runs the full pipeline on one GPU.

Usage:
    # Local, 4 GPUs
    uv run python scripts/prepare_fine_web_parallel.py \
        --output_dir=/work/hdd/bfaq/jlyu3/lcm/preprocessed_data \
        --num_samples=9200000 --num_shards=40

    # SLURM cluster (private dataset: set HF_TOKEN or --hf_token)
    export HF_TOKEN=your_token  # or: --hf_token=your_token
    uv run python scripts/prepare_fine_web_parallel.py \
        --output_dir=/work/hdd/bfaq/jlyu3/lcm/preprocessed_data \
        --num_samples=9200000 --num_shards=40 \
        --use_slurm=True --slurm_partition=gpuA100x4 --slurm_account=bfaq-delta-gpu

    # Tiny test (single GPU)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_fine_web_parallel.py \
        --output_dir=output/fine_web_test_parallel --num_samples=20 --num_shards=1
"""

import os

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/work/hdd/bfaq/jlyu3/lcm/hf_cache"
if "HF_DATASETS_CACHE" not in os.environ:
    os.environ["HF_DATASETS_CACHE"] = "/work/hdd/bfaq/jlyu3/lcm/hf_cache/datasets"

import asyncio
from pathlib import Path

import fire
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.partitioned_data_mapper import stopes_data_mapper
from stopes.utils.sharding.abstract_shards import BatchFormat
from stopes.utils.sharding.hf_shards import HFInputConfig
from stopes.utils.sharding.parquet_shards import ParquetOutputConfig

from lcm.datasets.fineweb_pipeline import FineWebPipeline, FineWebPipelineConfig


def prepare_fine_web_parallel(
    output_dir: str = "output/fine_web",
    num_samples: int = 100000,
    num_shards: int = 4,
    batch_size: int = 50,
    max_sentence_length: int = 256,
    train_ratio: float = 0.8,
    seed: int = 42,
    enable_unk_filter: bool = True,
    use_slurm: bool = False,
    slurm_partition: str = "gpuA100x4",
    slurm_account: str = "bfaq-delta-gpu",
    cache_dir: str = None,
    hf_token: str = None,
):
    """
    Multi-GPU parallel FineWeb-Edu preprocessing via stopes.

    Args:
        output_dir:  Directory for output parquet files.
        num_samples: Number of HF dataset rows to process.
        num_shards:  Number of parallel GPU jobs (set to available GPUs
                     or higher -- stopes will queue excess shards).
        batch_size:  Rows per stopes batch sent to FineWebPipeline.
        max_sentence_length: Max chars per sentence before truncation.
        train_ratio: Fraction assigned to the "train" split.
        seed:        Random seed for deterministic split assignment.
        enable_unk_filter: Drop sentences containing SONAR UNK tokens.
        use_slurm:   Submit jobs via SLURM instead of local multiprocessing.
        slurm_partition: SLURM partition name (ignored when use_slurm=False).
        slurm_account:   SLURM account name (ignored when use_slurm=False).
        cache_dir:   Path for stopes job cache.  When set, completed shards
                     are recorded here; rerunning the same command skips them
                     (shard-level resume).  Default None = no caching.
        hf_token:    HuggingFace access token for private datasets (e.g. LGVamper/fineweb-edu-20B).
                     If None, uses HF_TOKEN env var. Required for private/gated repos.

    Data-volume reference (FineWeb-Edu-20B):
        -   100K samples  ~  0.5B tokens  (num_shards=4)
        -   920K samples  ~  4.6B tokens  (num_shards=10)
        - 9.2M  samples  ~ 46B  tokens   (num_shards=40-80)

    The output directory will contain multiple parquet files produced by
    stopes.  The LCM training dataloader can read them directly -- no
    merge step is required.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FineWeb-Edu Multi-GPU Parallel Preprocessing (stopes)")
    print("=" * 80)
    print(f"  Output dir:   {output_dir}")
    print(f"  Samples:      {num_samples:,}")
    print(f"  Shards:       {num_shards}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Sentence max: {max_sentence_length}")
    print(f"  UNK filter:   {enable_unk_filter}")
    print(f"  Train ratio:  {train_ratio}")
    print(f"  SLURM:        {use_slurm}")
    if use_slurm:
        print(f"  Partition:    {slurm_partition}")
        print(f"  Account:      {slurm_account}")
    print(f"  Cache dir:    {cache_dir or '(disabled)'}")
    print(f"  HF token:     {'(set)' if (hf_token or os.environ.get('HF_TOKEN')) else '(not set)'}")
    print(f"  Per shard:    ~{num_samples // num_shards:,} samples")
    print("=" * 80)

    # Resolve HF token: CLI arg > env var. Set in env so local workers inherit it.
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    pipeline_config = FineWebPipelineConfig(
        max_sentence_length=max_sentence_length,
        train_ratio=train_ratio,
        seed=seed,
        enable_unk_filter=enable_unk_filter,
        sonar_batch_size=batch_size,
    )

    input_config = HFInputConfig(
        input_file="LGVamper/fineweb-edu-20B",
        split=f"train[0:{num_samples}]",
        num_shards=num_shards,
        batch_format=BatchFormat.ARROW,
        batch_size=batch_size,
    )

    output_config = ParquetOutputConfig(
        output_path,
        keep_same_partitioning=False,
        row_group_size=200,
        batch_size=200,
    )

    req = Requirements(
        mem_gb=120,
        gpus_per_node=1,
        cpus_per_task=10,
        timeout_min=7 * 24 * 60,
    )

    if use_slurm:
        slurm_setup = [
            f'export HF_HOME={os.environ["HF_HOME"]}',
            f'export HF_DATASETS_CACHE={os.environ["HF_DATASETS_CACHE"]}',
        ]
        if token:
            # Escape single quotes for safe shell export
            escaped = token.replace("'", "'\"'\"'")
            slurm_setup.append(f"export HF_TOKEN='{escaped}'")
        launcher_config = {
            "cluster": "slurm",
            "cache": cache_dir,
            "update_parameters": {
                "slurm_partition": slurm_partition,
                "slurm_account": slurm_account,
                "slurm_setup": slurm_setup,
            },
        }
        print(f"\nSubmitting {num_shards} SLURM jobs ...")
    else:
        launcher_config = {
            "cluster": "local",
            "cache": cache_dir,
        }
        print(f"\nLaunching {num_shards} local jobs ...")

    launcher = Launcher(**launcher_config)

    stopes_wrapped = stopes_data_mapper(req, {"name": "prep_fineweb_parallel"})(
        FineWebPipeline
    )
    stopes_module = stopes_wrapped(input_config, output_config, pipeline_config)

    print("Processing started ...")
    print("=" * 80)

    asyncio.run(launcher.schedule(stopes_module))

    print("\n" + "=" * 80)
    print("All shards complete.")
    print("=" * 80)
    print(f"  Output: {output_dir}")
    print(f"  Samples processed: {num_samples:,}")

    if use_slurm:
        print("\nMonitor SLURM jobs:  squeue -u $USER")
        print(f"View logs:  tail -f executor_logs/prep_fineweb_parallel/*.err")

    print("\nNext steps:")
    print(f"  1. Verify output:  ls -lh {output_dir}/")
    print("  2. (Optional) Pack:  python scripts/pack_parquet.py ...")
    print("  3. Update datacards.yaml  parquet_path -> this output dir")
    print("  4. Start training")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(prepare_fine_web_parallel)
