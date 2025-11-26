#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Multi-GPU Parallel C4 Data Preparation
支持多GPU并行处理，大幅加速数据准备
"""

import os
os.environ["HF_HOME"] = "/work/nvme/bfaq/xlin5/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/work/nvme/bfaq/xlin5/hf_cache/datasets"

import asyncio
from pathlib import Path
import fire

from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.partitioned_data_mapper import stopes_data_mapper
from stopes.modules.preprocess.sonar_text_embedding import (
    LangColumnConfig,
    SonarTextEmbedderConfig,
)
from stopes.utils.sharding.abstract_shards import BatchFormat
from stopes.utils.sharding.hf_shards import HFInputConfig
from stopes.utils.sharding.parquet_shards import ParquetOutputConfig

from lcm.datasets.sentence_splitter_pipeline import (
    FullPipeline,
    FullPipelineConfig,
    SentenceSplitterConfig,
)


def prepare_c4_parallel(
    output_dir: str = "output/c4_10b",
    num_samples: int = 2000000,
    num_shards: int = 64,
    batch_size: int = 50,
    use_slurm: bool = False,
    slurm_partition: str = "gpuA100x4",
    slurm_account: str = "bfaq-delta-gpu",
):
    """
    多GPU并行处理C4数据集（用于大规模数据）
    
    参数:
        output_dir: 输出目录
        num_samples: 样本数量（约2M样本 ≈ 10B tokens）
        num_shards: 分片数量（越多并行度越高）
        batch_size: 每个shard的批次大小
        use_slurm: 是否使用SLURM提交作业
        slurm_partition: SLURM分区
        slurm_account: SLURM账户
    
    数据量参考：
        - 100K样本 ≈ 0.5B tokens (num_samples=100000, num_shards=16)
        - 500K样本 ≈ 2.5B tokens (num_samples=500000, num_shards=32)
        - 2M样本 ≈ 10B tokens (num_samples=2000000, num_shards=64)
        - 20M样本 ≈ 100B tokens (num_samples=20000000, num_shards=128)
    
    使用示例:
    
    # 1. 本地多GPU（如果在gpua072/gpua081上有4个GPU）
    python scripts/prepare_c4_parallel.py \
        --output_dir=output/c4_10b \
        --num_samples=2000000 \
        --num_shards=64 \
        --use_slurm=False
    
    # 2. SLURM集群（推荐，大规模数据）
    python scripts/prepare_c4_parallel.py \
        --output_dir=output/c4_10b \
        --num_samples=2000000 \
        --num_shards=64 \
        --use_slurm=True \
        --slurm_partition=gpuA100x4 \
        --slurm_account=bfaq-delta-gpu
    
    时间估算（64个GPU并行）：
        - 2M样本: 6-12小时
        - 20M样本: 2-4天
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 C4数据集多GPU并行处理")
    print("=" * 80)
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 样本数量: {num_samples:,}")
    print(f"🔢 分片数量: {num_shards}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🖥️  使用SLURM: {use_slurm}")
    if use_slurm:
        print(f"📍 SLURM分区: {slurm_partition}")
        print(f"👤 SLURM账户: {slurm_account}")
    print("=" * 80)
    
    # 估算tokens和时间
    estimated_tokens = num_samples * 5000  # 保守估计，每样本5000 tokens
    print(f"\n📊 预计处理:")
    print(f"   - 样本数: {num_samples:,}")
    print(f"   - Tokens: ~{estimated_tokens/1e9:.1f}B")
    print(f"   - 每个shard: ~{num_samples//num_shards:,} 样本")
    
    if use_slurm:
        print(f"\n⏱️  预计时间（{num_shards}个GPU并行）:")
        hours = (num_samples / num_shards) * 0.01 / 60  # 粗略估计
        print(f"   - {hours:.1f} 小时")
    else:
        print(f"\n⏱️  预计时间（本地模式）:")
        print(f"   - 取决于可用GPU数量")
        print(f"   - 建议使用SLURM获得最佳性能")
    
    print("=" * 80)
    print()
    
    # 句子分割器配置
    splitter_config = SentenceSplitterConfig(
        columns=["text"],
        model_name="sat-3l",
        verbose=True,
        sentence_threshold=0.2,
        max_sentence_len=256,
    )
    
    # SONAR编码器配置
    sonar_encoder_config = SonarTextEmbedderConfig(
        column_config=[
            LangColumnConfig("text_sentences", lang_value="eng_Latn")
        ],
        device="cuda",
    )
    
    # 完整pipeline配置
    full_config = FullPipelineConfig(
        splitter_config=splitter_config,
        sonar_encoder_config=sonar_encoder_config,
    )
    
    # 输入配置 - C4流式
    input_config = HFInputConfig(
        input_file="allenai/c4",
        data_dir="en",
        split=f"train[0:{num_samples}]",
        num_shards=num_shards,
        batch_format=BatchFormat.ARROW,
        batch_size=batch_size,
    )
    
    # 输出配置
    output_config = ParquetOutputConfig(
        output_path,
        keep_same_partitioning=False,
        row_group_size=200,
        batch_size=200,
    )
    
    # 资源需求（每个shard）
    req = Requirements(
        mem_gb=120,
        gpus_per_node=1,
        cpus_per_task=10,
        timeout_min=7 * 24 * 60  # 7天
    )
    
    # Launcher配置
    if use_slurm:
        launcher_config = {
            "cluster": "slurm",
            "cache": None,
            "update_parameters": {
                "slurm_partition": slurm_partition,
                "slurm_account": slurm_account,
                "slurm_setup": [
                    f'export HF_HOME={os.environ["HF_HOME"]}',
                    f'export HF_DATASETS_CACHE={os.environ["HF_DATASETS_CACHE"]}',
                ]
            }
        }
        print(f"📤 提交 {num_shards} 个SLURM作业...")
    else:
        launcher_config = {
            "cluster": "local",
            "cache": None,
        }
        print(f"🖥️  启动本地处理（{num_shards} 个任务）...")
    
    launcher = Launcher(**launcher_config)
    
    # 包装pipeline
    stopes_wrapped = stopes_data_mapper(req, {"name": "prep_c4_parallel"})(FullPipeline)
    stopes_module = stopes_wrapped(input_config, output_config, full_config)
    
    print("\n🚀 开始处理...")
    print("=" * 80)
    
    # 启动处理
    asyncio.run(launcher.schedule(stopes_module))
    
    print("\n" + "=" * 80)
    print("✅ 所有任务已提交/完成！")
    print("=" * 80)
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 总样本数: {num_samples:,}")
    
    if use_slurm:
        print("\n📝 监控SLURM作业:")
        print(f"   squeue -u $USER")
        print("\n📝 查看日志:")
        print(f"   tail -f executor_logs/prep_c4_parallel/*.err")
    
    print("\n📝 下一步:")
    print("1. 等待所有任务完成")
    print(f"2. 验证输出: ls -lh {output_dir}/")
    print("3. 更新 lcm/datacards/datacards.yaml")
    print("4. 开始训练")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(prepare_c4_parallel)


