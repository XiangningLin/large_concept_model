# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Streaming fine web Data Preparation Script
真正的流式下载和处理，边下载边处理，不需要下载完整数据集
"""

import os
# 使用大空间目录存储 Hugging Face 缓存（如果环境变量未设置则使用此默认值）
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/work/hdd/bfaq/jlyu3/lcm/hf_cache"
if "HF_DATASETS_CACHE" not in os.environ:
    os.environ["HF_DATASETS_CACHE"] = "/work/hdd/bfaq/jlyu3/lcm/hf_cache/datasets"

from pathlib import Path
import sys
import json
import signal
import atexit
from typing import Optional
import torch
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sentence_splitter import SentenceSplitter
from stopes.utils.arrow_utils import nested_numpy_to_pyarrow
from timer import DistributedTimer

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

# we use s3 as our dataset key, but we actually don't have s3, it is just a placeholder
DATASET_NAME = "fine_web_edu"
CLUSTER_NAME = "s3"

def prepare_fine_web(
    output_dir: str = "output/fine_web",
    num_samples = None,  # int | None: 要处理的样本数量，None 表示处理整个数据集
    start_index: int = 0,
    batch_size: int = 10,
    max_sentence_length: int = 256,
    add_split_column: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42,
    checkpoint_interval: int = 5000,  # 每处理多少个样本保存一个检查点文件
    use_wandb: bool = False,  # 是否启用 wandb 监控
    wandb_project: str = "lcm_data_preparation",  # wandb 项目名
    wandb_run_name: Optional[str] = None,  # wandb run 名称，None 则自动生成
):
    """
    流式处理fine web数据集
    
    参数:
        output_dir: 输出目录
        num_samples: 从 start_index 开始要处理的样本数量，None 表示处理整个数据集
                    终止点为 start_index + num_samples（在数据集索引空间中）
        start_index: 数据集中的起始索引，从第 start_index 个样本开始处理（默认: 0）
        batch_size: 批处理大小（用于SONAR编码）
        max_sentence_length: 最大句子长度
        add_split_column: 是否添加 split 列用于 train/validation 分区（默认: True）
        train_ratio: 训练数据比例，当 add_split_column=True 时使用（默认: 0.8，即 80% 训练，20% 验证）
        seed: 随机种子，用于 split 的可重现性（默认: 42）
        checkpoint_interval: 每处理多少个样本保存一个检查点文件（默认: 50000）
        
    注意:
        - 如果从 start_index 到 start_index + num_samples 之间有空样本（空文本或无法分句），
          实际处理的有效样本数可能少于 num_samples，但会处理到索引 start_index + num_samples 为止
    
    使用示例:
        # 100条样本（5-10分钟）
        python scripts/prepare_fine_web.py --num_samples=100
        
        # 1000条样本（30-60分钟）
        python scripts/prepare_fine_web.py --num_samples=1000 --output_dir=output/fine_web_1k
        
        # 10000条样本（4-6小时）
        python scripts/prepare_fine_web.py --num_samples=10000 --output_dir=output/fine_web_10k
        
        # 处理整个数据集（num_samples=None）
        python scripts/prepare_fine_web.py --num_samples=None

        错误报告：
        我们遇到的主要错误是pyarrow在读取我们的parquet文件时，由于LCM的训练approach硬性要求sonar embedding为
        list<fixed_size_list[1024]>（即固定长度列表）格式，而我们的是可变长度列表。
        我们通过部署stopes的nested_numpy_to_pyarrow函数将可变长度列表转换为固定长度列表，从而解决了这个问题。
        见第270行代码：text_sentences_sonar_emb_pa = nested_numpy_to_pyarrow(all_embeddings)。
    """
    
    timer_root = Path(output_dir)
    timer = DistributedTimer("prepare_fine_web", root_dir=timer_root)
    timer.start()
    try:
        return _prepare_fine_web_impl(
            output_dir=output_dir,
            num_samples=num_samples,
            start_index=start_index,
            batch_size=batch_size,
            max_sentence_length=max_sentence_length,
            add_split_column=add_split_column,
            train_ratio=train_ratio,
            seed=seed,
            checkpoint_interval=checkpoint_interval,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )
    finally:
        timer.stop()


def _prepare_fine_web_impl(
    output_dir: str = "output/fine_web",
    num_samples = None,  # int | None: 要处理的样本数量，None 表示处理整个数据集
    start_index: int = 0,
    batch_size: int = 10,
    max_sentence_length: int = 256,
    add_split_column: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42,
    checkpoint_interval: int = 50000,  # 每处理多少个样本保存一个检查点文件
    use_wandb: bool = False,
    wandb_project: str = "lcm_data_preparation",
    wandb_run_name: Optional[str] = None,
):
    # 验证 checkpoint_interval
    if checkpoint_interval <= 0:
        raise ValueError(f"checkpoint_interval 必须 > 0，收到: {checkpoint_interval}")
    
    # 处理 num_samples 的默认值和 None 情况
    # 支持字符串 "None"、"none" 或 Python None
    if num_samples is None or (isinstance(num_samples, str) and num_samples.lower() in ["none", "all"]):
        num_samples = 100  # 默认值（实际不会使用，因为 process_all=True）
        process_all = True
    else:
        # 如果是字符串，尝试转换为整数
        if isinstance(num_samples, str):
            try:
                num_samples = int(num_samples)
            except ValueError:
                raise ValueError(f"num_samples 必须是整数或 None，收到: {num_samples}")
        process_all = False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata_file = output_path / ".progress_metadata.json"
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 检查点功能已启用：每 {checkpoint_interval:,} 个样本保存一个检查点文件")
    print(f"📁 检查点目录: {checkpoint_dir}")
    
    # 🔄 智能断点恢复：检查检查点文件并尝试从断点继续
    existing_samples = 0
    resume_from_index = None
    checkpoint_files = []  # 已存在的检查点文件列表
    next_checkpoint_number = 0  # 下一个检查点编号
    
    # 检查检查点文件
    if checkpoint_dir.exists():
        checkpoint_files = sorted(checkpoint_dir.glob("data_checkpoint_*.parquet"))
        if checkpoint_files:
            print("=" * 80)
            print("🔄 检测到检查点文件，尝试从检查点恢复...")
            print("=" * 80)
            print(f"📁 找到 {len(checkpoint_files)} 个检查点文件")
            
            # 统计已存在的样本数
            total_checkpoint_samples = 0
            for ckpt_file in checkpoint_files:
                try:
                    ckpt_table = pq.read_table(ckpt_file)
                    total_checkpoint_samples += len(ckpt_table)
                    print(f"   - {ckpt_file.name}: {len(ckpt_table):,} 个样本")
                except Exception as e:
                    print(f"   ⚠️  读取检查点文件 {ckpt_file.name} 失败: {e}")
            
            if total_checkpoint_samples > 0:
                existing_samples = total_checkpoint_samples
                print(f"✅ 从检查点恢复: 总计 {existing_samples:,} 个样本")
                
                # 确定下一个检查点编号
                if checkpoint_files:
                    # 从最后一个检查点文件名提取编号
                    last_ckpt_name = checkpoint_files[-1].name
                    try:
                        # 格式: data_checkpoint_<number>.parquet
                        last_number = int(last_ckpt_name.replace("data_checkpoint_", "").replace(".parquet", ""))
                        next_checkpoint_number = last_number + 1
                    except:
                        next_checkpoint_number = len(checkpoint_files)
                
                # 从 metadata 获取恢复索引
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        last_index = metadata.get('last_processed_index')
                        if last_index is not None:
                            resume_from_index = last_index + 1
                            print(f"📋 从 metadata 获取恢复索引: {resume_from_index:,}")
                    except Exception as e:
                        print(f"   ⚠️  读取 metadata 失败: {e}")
                
                # 如果没有 metadata，估算恢复索引
                if resume_from_index is None:
                    resume_from_index = start_index + existing_samples
                    print(f"📊 估算恢复索引: {resume_from_index:,}")
                
                print("=" * 80)
    
    # 如果检查点恢复失败，检查是否有旧的输出文件（兼容性处理）
    if resume_from_index is None and (output_path / "data.parquet").exists():
        print("=" * 80)
        print("⚠️  检测到旧的输出文件 data.parquet（已弃用）")
        print("   请使用 checkpoints/ 目录中的检查点文件")
        print("=" * 80)
    
    # 保存原始参数（用于 metadata）
    original_start_index = start_index
    original_num_samples = num_samples
    
    # 如果从断点恢复，更新 start_index
    if resume_from_index is not None:
        start_index = resume_from_index
    
    print("=" * 80)
    print("🚀 fine web 流式数据处理")
    print("=" * 80)
    print(f"📁 输出目录: {output_dir}")
    if process_all:
        print(f"📊 样本范围: {start_index:,} - 全部（处理整个数据集）")
        print(f"📊 样本数量: 全部")
    else:
        print(f"📊 样本范围: {start_index:,} - {start_index + num_samples:,}")
        print(f"📊 样本数量: {num_samples:,}")
    print(f"🔢 批次大小: {batch_size}")
    print(f"📏 最大句长: {max_sentence_length}")
    print("=" * 80)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("⚠️  未检测到GPU，使用CPU（会很慢）")
        device = torch.device("cpu")
    
    # 初始化 wandb（如果启用）
    wandb_run = None
    if use_wandb:
        if not has_wandb:
            print("⚠️  wandb 未安装，跳过 wandb 监控")
        else:
            if wandb_run_name is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wandb_run_name = f"prepare_fine_web_{Path(output_dir).name}_{timestamp}"
            
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                dir=output_path / "wandb",
                resume="allow",
                config={
                    "start_index": original_start_index,
                    "num_samples": original_num_samples if not process_all else None,
                    "process_all": process_all,
                    "batch_size": batch_size,
                    "max_sentence_length": max_sentence_length,
                    "checkpoint_interval": checkpoint_interval,
                    "output_dir": str(output_dir),
                },
            )
            print(f"✅ WandB 监控已启用: {wandb_project}/{wandb_run_name}")
    
    print("\n🔄 初始化模型...")
    
    # 初始化句子分割器
    splitter = SentenceSplitter(language='en')
    
    # 初始化SONAR编码器
    print("   - 加载SONAR模型...")
    sonar_pipeline = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    )
    print("✅ 模型加载完成\n")
    
    # 流式加载fine web数据集
    if process_all:
        print(f"📥 开始流式下载fine web数据（处理整个数据集）...")
    else:
        print(f"📥 开始流式下载fine web数据 (从索引 {start_index} 开始，处理 {num_samples} 条)...")
    print("   (这是真正的流式，只下载需要的数据)\n")
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True  # 🔥 流式下载！
    )
    
    # 处理状态
    processed = 0
    last_processed_index = start_index - 1  # 记录最后处理的索引
    current_checkpoint_data = []  # 当前检查点的数据（累积到 checkpoint_interval）
    
    # 保存检查点文件的函数
    def save_checkpoint(checkpoint_data, checkpoint_num):
        """保存检查点文件"""
        if not checkpoint_data:
            return
        
        try:
            # 添加 split 列（如果需要）
            if add_split_column:
                import random
                random.seed(seed)
                # 为检查点数据分配 split（基于全局索引）
                total_before = existing_samples + (checkpoint_num * checkpoint_interval)
                for i, d in enumerate(checkpoint_data):
                    global_idx = total_before + i
                    # 使用全局索引的哈希值来分配 split（确保一致性）
                    if hash((global_idx, seed)) % 100 < int(train_ratio * 100):
                        d['split'] = 'train'
                    else:
                        d['split'] = 'validation'
            else:
                for d in checkpoint_data:
                    d['split'] = 'train'  # 默认值
            
            # 转换为 PyArrow table
            text_sentences_list = [d['text_sentences'] for d in checkpoint_data]
            all_embeddings = [d['text_sentences_sonar_emb'] for d in checkpoint_data]
            text_sentences_sonar_emb_pa = nested_numpy_to_pyarrow(all_embeddings)
            
            checkpoint_table = pa.table({
                'text_sentences': text_sentences_list,
                'text_sentences_sonar_emb': text_sentences_sonar_emb_pa,
                'url': [d['url'] for d in checkpoint_data],
                'timestamp': [d['timestamp'] for d in checkpoint_data],
                'split': [d.get('split', 'train') for d in checkpoint_data],
            })
            
            # 保存检查点文件
            checkpoint_file = checkpoint_dir / f"data_checkpoint_{checkpoint_num:06d}.parquet"
            pq.write_table(checkpoint_table, checkpoint_file)
            print(f"\n💾 检查点已保存: {checkpoint_file.name} ({len(checkpoint_data):,} 个样本)")
            
        except Exception as e:
            print(f"⚠️  保存检查点失败: {e}")
    
    # 保存 metadata 的函数（使用闭包保存原始参数）
    def save_metadata():
        """保存处理进度到 metadata 文件"""
        # 使用原始参数（在函数定义时捕获）
        metadata = {
            'start_index': original_start_index,  # 原始 start_index
            'num_samples': original_num_samples if not process_all else None,  # 原始 num_samples
            'process_all': process_all,
            'processed_samples': processed + existing_samples,
            'last_processed_index': last_processed_index,
            'add_split_column': add_split_column,
            'train_ratio': train_ratio,
            'seed': seed,
            'checkpoint_interval': checkpoint_interval,
            'checkpoint_count': next_checkpoint_number,
        }
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  保存 metadata 失败: {e}")
    
    # 注册退出时保存 metadata
    atexit.register(save_metadata)
    
    # 处理中断信号，保存进度
    def signal_handler(signum, frame):
        print(f"\n\n⚠️  收到中断信号 ({signum})，保存当前进度...")
        if current_checkpoint_data:
            print(f"   保存当前检查点 ({len(current_checkpoint_data)} 个样本)...")
            save_checkpoint(current_checkpoint_data, next_checkpoint_number)
        save_metadata()
        if current_checkpoint_data:
            print(f"   已处理 {processed} 个新样本")
            print(f"   检查点文件已保存到: {checkpoint_dir}")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 使用tqdm显示进度（如果处理全部数据，total=None 表示未知总数）
    total_expected = None if process_all else num_samples
    with tqdm(total=total_expected, desc="处理进度", unit="样本", initial=existing_samples) as pbar:
        def _encode_batch(batch_texts, batch_originals):
            nonlocal processed, last_processed_index, current_checkpoint_data, next_checkpoint_number
            if not batch_texts:
                return

            flat_sentences = []
            doc_offsets = []
            offset = 0
            for sents in batch_texts:
                length = len(sents)
                doc_offsets.append((offset, length))
                flat_sentences.extend(sents)
                offset += length

            if not flat_sentences:
                return

            doc_embeddings = sonar_pipeline.predict(
                flat_sentences,
                source_lang="eng_Latn"
            )

            if not isinstance(doc_embeddings, torch.Tensor):
                raise ValueError(
                    f"Expected torch.Tensor from SONAR predict, got {type(doc_embeddings)}"
                )
            if len(doc_embeddings.shape) != 2:
                raise ValueError(
                    f"Expected 2D tensor (num_sentences, embedding_dim), got shape {doc_embeddings.shape}"
                )

            flat_embeddings = doc_embeddings.cpu()

            for i, (start, length) in enumerate(doc_offsets):
                if length == 0:
                    continue

                emb_numpy = flat_embeddings[start:start + length].numpy()
                dataset_idx = batch_originals[i].get('dataset_index', start_index + processed)
                sample_data = {
                    'text_sentences': batch_texts[i],
                    'text_sentences_sonar_emb': emb_numpy,
                    'url': batch_originals[i]['url'],
                    'timestamp': batch_originals[i]['timestamp'],
                }

                processed += 1
                last_processed_index = dataset_idx
                pbar.update(1)

                current_checkpoint_data.append(sample_data)
                if len(current_checkpoint_data) >= checkpoint_interval:
                    save_checkpoint(current_checkpoint_data, next_checkpoint_number)
                    current_checkpoint_data = []
                    next_checkpoint_number += 1
                    # 检查点保存时记录进度
                    if wandb_run is not None:
                        log_dict = {"progress/processed_samples": processed}
                        if not process_all:
                            percentage = (processed / num_samples) * 100
                            log_dict["progress/percentage"] = percentage
                        wandb_run.log(log_dict, step=processed, commit=True)

                if processed % 100 == 0:
                    save_metadata()
                    # 记录进度到 wandb
                    if wandb_run is not None:
                        log_dict = {"progress/processed_samples": processed}
                        if not process_all:
                            percentage = (processed / num_samples) * 100
                            log_dict["progress/percentage"] = percentage
                        wandb_run.log(log_dict, step=processed)

        batch_texts = []
        batch_originals = []
        
        for idx, sample in enumerate(dataset):
            # 跳过start_index之前的样本
            if idx < start_index:
                continue
            # 如果不是处理全部数据，检查是否达到终止点（start_index + num_samples）
            if not process_all and idx >= start_index + num_samples:
                break
            
            text = sample['text'].strip()
            if not text:
                last_processed_index = idx  # 记录索引，即使跳过了
                continue
            
            # 分句
            sentences = splitter.split(text)
            if not sentences:
                last_processed_index = idx  # 记录索引，即使跳过了
                continue
            
            # 截断过长的句子
            sentences = [s[:max_sentence_length] for s in sentences]
            
            batch_texts.append(sentences)
            batch_originals.append({
                'url': sample.get('url', ''),
                'timestamp': sample.get('timestamp', ''),
                'dataset_index': idx,  # 保存原始索引，用于追踪
            })
            
            # 批量编码（将一个 batch 展平后统一编码，再切回每个 sample）
            if len(batch_texts) >= batch_size:
                _encode_batch(batch_texts, batch_originals)
                batch_texts = []
                batch_originals = []
        
        # 处理剩余的数据（也使用批处理）
        # 注意：循环可能因为达到终止点而结束，此时 batch_texts 中的样本索引都在有效范围内
        if batch_texts:
            _encode_batch(batch_texts, batch_originals)
    
    # 处理剩余的检查点数据
    if current_checkpoint_data:
        print(f"\n💾 保存最后一个检查点（剩余 {len(current_checkpoint_data)} 个样本）...")
        save_checkpoint(current_checkpoint_data, next_checkpoint_number)
        current_checkpoint_data = []
        next_checkpoint_number += 1
    
    # 统计所有检查点文件
    all_checkpoint_files = sorted(checkpoint_dir.glob("data_checkpoint_*.parquet"))
    total_samples = 0
    if all_checkpoint_files:
        print(f"\n📊 检查点文件统计:")
        print(f"   找到 {len(all_checkpoint_files)} 个检查点文件")
        for ckpt_file in all_checkpoint_files:
            try:
                ckpt_table = pq.read_table(ckpt_file)
                total_samples += len(ckpt_table)
                print(f"   - {ckpt_file.name}: {len(ckpt_table):,} 个样本")
            except Exception as e:
                print(f"   ⚠️  读取检查点文件 {ckpt_file.name} 失败: {e}")
    
    # 更新并保存 metadata
    save_metadata()
    
    # 记录最终进度到 wandb
    if wandb_run is not None:
        log_dict = {"progress/processed_samples": processed}
        if not process_all:
            percentage = (processed / num_samples) * 100
            log_dict["progress/percentage"] = percentage
        wandb_run.log(log_dict, step=processed, commit=True)
        wandb_run.finish()
    
    print("\n" + "=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print(f"📁 检查点目录: {checkpoint_dir}")
    print(f"📊 检查点文件数: {len(all_checkpoint_files)}")
    print(f"📊 总样本数: {total_samples:,}")
    print(f"📊 本次处理样本: {processed}")
    print("\n📝 提示:")
    print("   训练时，请将 datacard 中的 parquet_path 指向 checkpoints/ 目录")
    print("   例如: parquet_path: s3: \"output/fine_web/checkpoints\"")
    print("   多卡运行时，datacard 将在所有任务完成后统一更新")
    print("   如需手动更新，请运行:")
    print(f"   python scripts/update_datacards.py --output_dir={output_dir} --dataset_name={DATASET_NAME}")
    print("\n📝 提示:")
    print("   多卡运行时，datacard 将在所有任务完成后统一更新")
    print("   如需手动更新，请运行:")
    print(f"   python scripts/update_datacards.py --output_dir={output_dir} --dataset_name={DATASET_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    import fire
    fire.Fire(prepare_fine_web)
