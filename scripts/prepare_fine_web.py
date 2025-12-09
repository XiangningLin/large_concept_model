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
import torch
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sentence_splitter import SentenceSplitter
from stopes.utils.arrow_utils import nested_numpy_to_pyarrow

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
    checkpoint_interval: int = 50000,  # 每处理多少个样本保存一个检查点文件
):
    """
    流式处理fine web数据集
    
    参数:
        output_dir: 输出目录
        num_samples: 要处理的样本数量，None 表示处理整个数据集（默认: 100）
        start_index: 起始索引（用于多GPU并行）
        batch_size: 批处理大小（用于SONAR编码）
        max_sentence_length: 最大句子长度
        add_split_column: 是否添加 split 列用于 train/validation 分区（默认: True）
        train_ratio: 训练数据比例，当 add_split_column=True 时使用（默认: 0.8，即 80% 训练，20% 验证）
        seed: 随机种子，用于 split 的可重现性（默认: 42）
        checkpoint_interval: 每处理多少个样本保存一个检查点文件（默认: 1000）。设置为 0 或负数可禁用检查点
    
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
    
    output_file = output_path / "data.parquet"
    metadata_file = output_path / ".progress_metadata.json"
    checkpoint_dir = output_path / "checkpoints"
    
    # 启用检查点功能（如果 checkpoint_interval > 0）
    use_checkpoints = checkpoint_interval > 0
    if use_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 检查点功能已启用：每 {checkpoint_interval:,} 个样本保存一个检查点文件")
        print(f"📁 检查点目录: {checkpoint_dir}")
    else:
        print("⚠️  检查点功能已禁用（checkpoint_interval <= 0）")
    
    # 🔄 智能断点恢复：检查检查点文件和输出文件并尝试从断点继续
    existing_data = None
    existing_samples = 0
    resume_from_index = None
    checkpoint_files = []  # 已存在的检查点文件列表
    next_checkpoint_number = 0  # 下一个检查点编号
    
    # 首先检查检查点文件（如果启用检查点功能）
    if use_checkpoints and checkpoint_dir.exists():
        checkpoint_files = sorted(checkpoint_dir.glob("data_checkpoint_*.parquet"))
        if checkpoint_files:
            print("=" * 80)
            print("🔄 检测到检查点文件，尝试从检查点恢复...")
            print("=" * 80)
            print(f"📁 找到 {len(checkpoint_files)} 个检查点文件")
            
            # 读取所有检查点文件
            checkpoint_tables = []
            total_checkpoint_samples = 0
            for ckpt_file in checkpoint_files:
                try:
                    ckpt_table = pq.read_table(ckpt_file)
                    checkpoint_tables.append(ckpt_table)
                    total_checkpoint_samples += len(ckpt_table)
                    print(f"   - {ckpt_file.name}: {len(ckpt_table):,} 个样本")
                except Exception as e:
                    print(f"   ⚠️  读取检查点文件 {ckpt_file.name} 失败: {e}")
            
            if checkpoint_tables:
                # 合并所有检查点
                existing_data = pa.concat_tables(checkpoint_tables)
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
    
    # 如果检查点恢复失败或未启用检查点，检查最终输出文件
    if existing_data is None and output_file.exists():
        print("=" * 80)
        print("🔄 检测到已存在的输出文件，尝试智能断点恢复...")
        print("=" * 80)
        print(f"📁 输出文件: {output_file}")
        
        try:
            # 读取已存在的文件
            existing_table = pq.read_table(output_file)
            existing_samples = len(existing_table)
            file_size_mb = output_file.stat().st_size / 1024 / 1024
            
            print(f"📊 已存在文件信息:")
            print(f"   - 样本数量: {existing_samples:,}")
            print(f"   - 文件大小: {file_size_mb:.2f} MB")
            
            # 读取 metadata（如果存在）
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"📋 找到进度元数据:")
                    print(f"   - 原始 start_index: {metadata.get('start_index', 'N/A')}")
                    print(f"   - 原始 num_samples: {metadata.get('num_samples', 'N/A')}")
                    print(f"   - 已处理样本数: {metadata.get('processed_samples', existing_samples)}")
                    print(f"   - 最后处理的索引: {metadata.get('last_processed_index', 'N/A')}")
                except Exception as e:
                    print(f"   ⚠️  读取 metadata 失败: {e}")
            
            # 判断是否可以恢复
            can_resume = False
            resume_reason = ""
            
            if process_all:
                # 处理全部数据时，如果文件存在且有一定大小（>1MB），认为可能完整
                if existing_samples >= num_samples and file_size_mb > 1.0:
                    can_resume = True
                    resume_reason = f"文件已存在且大小合理 ({file_size_mb:.2f} MB, {existing_samples:,} 样本)，跳过处理"
                else:
                    # 尝试从断点继续
                    last_index = metadata.get('last_processed_index', start_index + existing_samples - 1)
                    if last_index is not None and last_index >= start_index:
                        resume_from_index = last_index + 1
                        can_resume = True
                        resume_reason = f"从索引 {resume_from_index:,} 继续处理（已处理 {existing_samples:,} 个样本）"
                    else:
                        resume_reason = f"文件存在但可能不完整 (大小: {file_size_mb:.2f} MB)，将重新处理"
            else:
                # 处理指定数量时，检查样本数是否达到预期
                if existing_samples >= num_samples:
                    can_resume = True
                    resume_reason = f"文件已包含 {existing_samples:,} 个样本，达到预期 {num_samples:,}，跳过处理"
                else:
                    # 计算还需要处理多少样本
                    remaining_samples = num_samples - existing_samples
                    # 计算从哪个索引继续（基于已处理的样本数）
                    # 如果 metadata 中有 last_processed_index，使用它；否则估算
                    if 'last_processed_index' in metadata:
                        resume_from_index = metadata['last_processed_index'] + 1
                    else:
                        # 估算：假设已处理的样本是从 start_index 开始的连续样本
                        resume_from_index = start_index + existing_samples
                    
                    if resume_from_index >= start_index:
                        can_resume = True
                        resume_reason = f"从索引 {resume_from_index:,} 继续处理（已处理 {existing_samples:,}/{num_samples:,}，还需 {remaining_samples:,} 个样本）"
                        # 保存已存在的数据，后续会追加新数据
                        existing_data = existing_table
                    else:
                        resume_reason = f"文件只包含 {existing_samples:,} 个样本，未达到预期 {num_samples:,}，将重新处理"
            
            if can_resume:
                if resume_from_index is not None:
                    # 从断点继续处理
                    print(f"\n✅ 断点恢复成功！")
                    print(f"   {resume_reason}")
                    print(f"   📁 将追加到已存在的文件: {output_file}")
                    print(f"   🔄 从索引 {resume_from_index:,} 继续处理...")
                    print("=" * 80)
                    # 更新参数以从断点继续（保存原始参数用于 metadata）
                    # 注意：original_start_index 和 original_num_samples 会在后面定义时使用
                    # 这里先保存原始值
                    _saved_original_start_index = start_index
                    _saved_original_num_samples = num_samples
                    start_index = resume_from_index
                    if not process_all:
                        num_samples = remaining_samples  # 更新为还需要处理的样本数
                else:
                    # 文件已完整，直接返回
                    print(f"\n✅ 断点恢复成功！")
                    print(f"   {resume_reason}")
                    print(f"   📁 使用已存在的完整文件: {output_file}")
                    print("=" * 80)
                    return  # 直接返回，跳过处理
            else:
                print(f"\n⚠️  无法从断点恢复")
                print(f"   {resume_reason}")
                print(f"   🔄 将重新处理数据...")
                print("=" * 80)
                # 如果无法恢复，删除不完整的文件
                if existing_samples < num_samples:
                    print(f"   🗑️  删除不完整的文件以重新处理...")
                    output_file.unlink()
                    if metadata_file.exists():
                        metadata_file.unlink()
        except Exception as e:
            print(f"\n⚠️  读取已存在文件时出错: {e}")
            print(f"   🔄 将重新处理数据...")
            print("=" * 80)
    
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
        print(f"📥 开始流式下载fine web数据 (前 {num_samples} 条)...")
    print("   (这是真正的流式，只下载需要的数据)\n")
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True  # 🔥 流式下载！
    )
    
    # 收集处理后的数据
    all_data = []
    processed = 0
    last_processed_index = start_index - 1  # 记录最后处理的索引
    current_checkpoint_data = []  # 当前检查点的数据（累积到 checkpoint_interval）
    
    # 保存原始参数（用于 metadata）
    # 如果是从断点恢复，使用保存的原始值；否则使用当前值
    if 'resume_from_index' in locals() and resume_from_index is not None and '_saved_original_start_index' in locals():
        original_start_index = _saved_original_start_index
        original_num_samples = _saved_original_num_samples
    else:
        original_start_index = start_index
        original_num_samples = num_samples
    
    # 保存检查点文件的函数
    def save_checkpoint(checkpoint_data, checkpoint_num):
        """保存检查点文件"""
        if not use_checkpoints or not checkpoint_data:
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
            'checkpoint_interval': checkpoint_interval if use_checkpoints else None,
            'checkpoint_count': next_checkpoint_number if use_checkpoints else None,
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
        # 如果启用检查点且有未保存的数据，保存当前检查点
        if use_checkpoints and current_checkpoint_data:
            print(f"   保存当前检查点 ({len(current_checkpoint_data)} 个样本)...")
            save_checkpoint(current_checkpoint_data, next_checkpoint_number)
        save_metadata()
        if all_data or current_checkpoint_data:
            print(f"   已处理 {processed} 个新样本")
            if use_checkpoints:
                print(f"   检查点文件已保存到: {checkpoint_dir}")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 使用tqdm显示进度（如果处理全部数据，total=None 表示未知总数）
    total_expected = None if process_all else num_samples
    if existing_data is not None:
        total_expected = (total_expected or 0) + existing_samples
    with tqdm(total=total_expected, desc="处理进度", unit="样本", initial=existing_samples) as pbar:
        batch_texts = []
        batch_originals = []
        
        for idx, sample in enumerate(dataset):
            # 跳过start_index之前的样本
            if idx < start_index:
                continue
            # 如果不是处理全部数据，检查是否达到目标数量
            if not process_all and processed >= num_samples:
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
            
            # 批量编码
            if len(batch_texts) >= batch_size:
                # 处理这一批
                for i, sents in enumerate(batch_texts):
                    try:
                        # 编码句子
                        embeddings = sonar_pipeline.predict(
                            sents,
                            source_lang="eng_Latn"
                        )
                        
                        # 🔍 Validate embeddings format (only for first successfully processed sample)
                        if processed == 0:  # Only print for the first sample
                            print(f"\n🔍 [Validation] First sample embeddings info:", flush=True)
                            print(f"   - embeddings type: {type(embeddings)}", flush=True)
                            print(f"   - embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}", flush=True)
                            if hasattr(embeddings, 'cpu'):
                                emb_numpy = embeddings.cpu().numpy()
                                print(f"   - numpy array type: {type(emb_numpy)}", flush=True)
                                print(f"   - numpy array shape: {emb_numpy.shape}", flush=True)
                                print(f"   - numpy array dtype: {emb_numpy.dtype}", flush=True)
                                print(f"   - type after converting to list: {type(emb_numpy.tolist())}", flush=True)
                                print(f"   - list length: {len(emb_numpy.tolist())}", flush=True)
                                if len(emb_numpy.tolist()) > 0:
                                    print(f"   - list[0] type: {type(emb_numpy.tolist()[0])}", flush=True)
                                    if isinstance(emb_numpy.tolist()[0], list):
                                        print(f"   - list[0] length: {len(emb_numpy.tolist()[0])}", flush=True)
                            sys.stdout.flush()  # Force flush output buffer
                        
                        # 保存数据（记录当前处理的索引）
                        dataset_idx = batch_originals[i].get('dataset_index', start_index + processed)
                        
                        # 准备数据字典
                        sample_data = {
                            'text_sentences': sents,
                            'text_sentences_sonar_emb': embeddings.cpu().numpy(),
                            'url': batch_originals[i]['url'],
                            'timestamp': batch_originals[i]['timestamp'],
                        }
                        
                        processed += 1
                        last_processed_index = dataset_idx  # 使用数据集中的实际索引
                        pbar.update(1)
                        
                        # 如果启用检查点，将数据添加到当前检查点；否则直接添加到 all_data
                        if use_checkpoints:
                            current_checkpoint_data.append(sample_data)
                            
                            # 达到检查点间隔，保存检查点
                            if len(current_checkpoint_data) >= checkpoint_interval:
                                save_checkpoint(current_checkpoint_data, next_checkpoint_number)
                                # 将检查点数据添加到 all_data（用于最终合并）
                                all_data.extend(current_checkpoint_data)
                                current_checkpoint_data = []  # 清空当前检查点
                                next_checkpoint_number += 1
                        else:
                            # 未启用检查点，直接添加到 all_data
                            all_data.append(sample_data)
                        
                        # 每处理 100 个样本保存一次 metadata（防止意外中断丢失进度）
                        if processed % 100 == 0:
                            save_metadata()
                        
                    except Exception as e:
                        print(f"\n⚠️  Processing failed: {str(e)}")
                        continue
                
                batch_texts = []
                batch_originals = []
        
        # 处理剩余的数据
        if batch_texts and (process_all or processed < num_samples):
            for i, sents in enumerate(batch_texts):
                if not process_all and processed >= num_samples:
                    break
                try:
                    embeddings = sonar_pipeline.predict(
                        sents,
                        source_lang="eng_Latn"
                    )
                    
                    # 🔍 Validate embeddings format (remaining data)
                    if processed == 0 and len(all_data) == 0:  # Only print for the first sample
                        print(f"\n🔍 [Validation] Remaining data first sample embeddings info:", flush=True)
                        print(f"   - embeddings type: {type(embeddings)}", flush=True)
                        print(f"   - embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}", flush=True)
                        if hasattr(embeddings, 'cpu'):
                            emb_numpy = embeddings.cpu().numpy()
                            print(f"   - numpy array type: {type(emb_numpy)}", flush=True)
                            print(f"   - numpy array shape: {emb_numpy.shape}", flush=True)
                        sys.stdout.flush()  # Force flush output buffer
                    
                    # 保存数据（记录当前处理的索引）
                    dataset_idx = batch_originals[i].get('dataset_index', start_index + processed)
                    
                    # 准备数据字典
                    sample_data = {
                        'text_sentences': sents,
                        'text_sentences_sonar_emb': embeddings.cpu().numpy(),
                        'url': batch_originals[i]['url'],
                        'timestamp': batch_originals[i]['timestamp'],
                    }
                    
                    processed += 1
                    last_processed_index = dataset_idx  # 使用数据集中的实际索引
                    pbar.update(1)
                    
                    # 如果启用检查点，将数据添加到当前检查点；否则直接添加到 all_data
                    if use_checkpoints:
                        current_checkpoint_data.append(sample_data)
                        
                        # 达到检查点间隔，保存检查点
                        if len(current_checkpoint_data) >= checkpoint_interval:
                            save_checkpoint(current_checkpoint_data, next_checkpoint_number)
                            # 将检查点数据添加到 all_data（用于最终合并）
                            all_data.extend(current_checkpoint_data)
                            current_checkpoint_data = []  # 清空当前检查点
                            next_checkpoint_number += 1
                    else:
                        # 未启用检查点，直接添加到 all_data
                        all_data.append(sample_data)
                    
                    # 每处理 100 个样本保存一次 metadata（防止意外中断丢失进度）
                    if processed % 100 == 0:
                        save_metadata()
                    
                except Exception as e:
                    print(f"\n⚠️  Processing failed: {str(e)}")
                    continue
    
    # 如果启用检查点，处理剩余的检查点数据
    if use_checkpoints and current_checkpoint_data:
        print(f"\n💾 保存最后一个检查点（剩余 {len(current_checkpoint_data)} 个样本）...")
        save_checkpoint(current_checkpoint_data, next_checkpoint_number)
        all_data.extend(current_checkpoint_data)
        current_checkpoint_data = []
        next_checkpoint_number += 1
    
    # 如果启用检查点，从检查点文件合并数据（而不是使用 all_data）
    if use_checkpoints:
        print(f"\n🔄 合并所有检查点文件为最终输出...")
        # 读取所有检查点文件（包括已存在的和新保存的）
        all_checkpoint_files = sorted(checkpoint_dir.glob("data_checkpoint_*.parquet"))
        if all_checkpoint_files:
            print(f"   找到 {len(all_checkpoint_files)} 个检查点文件")
            checkpoint_tables = []
            for ckpt_file in all_checkpoint_files:
                try:
                    ckpt_table = pq.read_table(ckpt_file)
                    checkpoint_tables.append(ckpt_table)
                    print(f"   - {ckpt_file.name}: {len(ckpt_table):,} 个样本")
                except Exception as e:
                    print(f"   ⚠️  读取检查点文件 {ckpt_file.name} 失败: {e}")
            
            if checkpoint_tables:
                # 合并所有检查点
                new_table = pa.concat_tables(checkpoint_tables)
                print(f"   ✅ 合并完成，总计 {len(new_table):,} 个样本")
            else:
                # 如果没有检查点文件，使用 all_data（这种情况不应该发生）
                print("   ⚠️  没有找到检查点文件，使用内存中的数据")
                # 继续使用原来的逻辑处理 all_data
                new_table = None
        else:
            # 没有检查点文件，使用 all_data
            print("   ⚠️  没有找到检查点文件，使用内存中的数据")
            new_table = None
    else:
        # 未启用检查点，使用原来的逻辑
        new_table = None
    
    # 如果 new_table 为 None（未启用检查点或检查点合并失败），使用 all_data 创建 table
    if new_table is None:
        # 添加 split 列（如果启用）
        # 注意：如果是从断点恢复，需要基于总样本数（包括已存在的）来分配 split
        if add_split_column:
            import random
            random.seed(seed)
            
            # 计算总样本数（包括已存在的）
            total_existing = existing_samples if existing_data is not None else 0
            total_new = len(all_data)
            total_samples = total_existing + total_new
            
            # 如果是从断点恢复，需要确保 split 分配的一致性
            # 使用全局索引来分配 split，确保恢复后的一致性
            train_size = int(total_samples * train_ratio)
            
            # 随机打乱全局索引
            indices = list(range(total_samples))
            random.shuffle(indices)
            train_indices = set(indices[:train_size])
            
            # 只为新数据分配 split（已存在的数据应该已经有 split 了）
            for i in range(total_new):
                global_idx = total_existing + i
                if global_idx in train_indices:
                    all_data[i]['split'] = 'train'
                else:
                    all_data[i]['split'] = 'validation'
            
            new_train = sum(1 for d in all_data if d.get('split') == 'train')
            new_val = len(all_data) - new_train
            print(f"\n✅ Added split column to new data: {new_train} train, {new_val} validation")
            if total_existing > 0:
                print(f"   (Total: {total_samples:,} samples, {train_size:,} train, {total_samples - train_size:,} validation)")
        
        # Save as Parquet file
        print(f"\n💾 Saving data to {output_dir}/data.parquet ...")
        
        # 🔍 Validate data format (before conversion)
        if len(all_data) > 0:
            print(f"\n🔍 [Validation] Data format before conversion:")
            first_emb = all_data[0]['text_sentences_sonar_emb']
            print(f"   - all_data[0]['text_sentences_sonar_emb'] type: {type(first_emb)}")
            print(f"   - all_data[0]['text_sentences_sonar_emb'] shape: {first_emb.shape if hasattr(first_emb, 'shape') else 'N/A'}")
            print(f"   - type after converting to list: {type(first_emb.tolist())}")
            print(f"   - list length: {len(first_emb.tolist())}")
            if len(first_emb.tolist()) > 0:
                print(f"   - list[0] type: {type(first_emb.tolist()[0])}")
                if isinstance(first_emb.tolist()[0], (list, np.ndarray)):
                    print(f"   - list[0] length: {len(first_emb.tolist()[0])}")
        
        # Convert embeddings to fixed_size_list format (PyArrow compatible)
        text_sentences_list = [d['text_sentences'] for d in all_data]
        all_embeddings = [d['text_sentences_sonar_emb'] for d in all_data]  # List of numpy arrays
        text_sentences_sonar_emb_pa = nested_numpy_to_pyarrow(all_embeddings)  # Creates list<fixed_size_list[1024]>
        
        # 🔍 Validate converted format
        if len(all_embeddings) > 0:
            print(f"\n🔍 [Validation] Converted format:")
            print(f"   - text_sentences_sonar_emb_pa type: {text_sentences_sonar_emb_pa.type}", flush=True)
            print(f"   - Is value_type fixed_size_list? {pa.types.is_fixed_size_list(text_sentences_sonar_emb_pa.type.value_type) if pa.types.is_list(text_sentences_sonar_emb_pa.type) else False}", flush=True)
            if pa.types.is_list(text_sentences_sonar_emb_pa.type) and pa.types.is_fixed_size_list(text_sentences_sonar_emb_pa.type.value_type):
                print(f"   - Fixed size: {text_sentences_sonar_emb_pa.type.value_type.list_size}", flush=True)
            sys.stdout.flush()
        
        # 创建新数据的 table
        new_table = pa.table({
            'text_sentences': text_sentences_list,
            'text_sentences_sonar_emb': text_sentences_sonar_emb_pa,
            'url': [d['url'] for d in all_data],
            'timestamp': [d['timestamp'] for d in all_data],
            'split': [d.get('split', 'train') for d in all_data],  # Add split column, default to 'train'
        })
    else:
        # 如果使用检查点合并的 new_table，直接使用它（split 列已经在检查点中处理了）
        print(f"\n💾 使用检查点合并的数据，保存到 {output_dir}/data.parquet ...")
    
    # 如果存在旧数据且未使用检查点，合并新旧数据
    # 注意：如果使用检查点，existing_data 已经包含了所有检查点数据，new_table 也是从检查点合并的，所以不需要再合并
    if existing_data is not None and not use_checkpoints:
        print(f"\n🔄 合并新旧数据...")
        print(f"   - 已存在样本: {existing_samples:,}")
        print(f"   - 新处理样本: {len(new_table):,}")
        print(f"   - 总计: {existing_samples + len(new_table):,}")
        
        # 确保 schema 一致
        if existing_data.schema != new_table.schema:
            print("   ⚠️  Schema 不一致，尝试对齐...")
            # 对齐列的顺序和类型
            existing_data = existing_data.select(new_table.column_names)
        
        # 合并两个 table
        final_table = pa.concat_tables([existing_data, new_table])
        print(f"   ✅ 合并完成，总计 {len(final_table):,} 个样本")
    else:
        # 如果使用检查点，new_table 已经包含了所有数据（从检查点合并）
        # 如果不存在旧数据，直接使用 new_table
        final_table = new_table
        if use_checkpoints:
            print(f"   ✅ 检查点数据已合并，总计 {len(final_table):,} 个样本")
    
    # 🔍 Validate PyArrow table schema
    print(f"\n🔍 [Validation] PyArrow Table Schema:")
    print(f"   - Table column count: {len(final_table.column_names)}")
    print(f"   - Table row count: {len(final_table)}")
    for col_name in final_table.column_names:
        col_type = final_table[col_name].type
        print(f"   - {col_name}: {col_type}")
        if col_name == 'text_sentences_sonar_emb':
            print(f"     - is list type: {pa.types.is_list(col_type)}")
            if pa.types.is_list(col_type):
                print(f"     - value type: {col_type.value_type}")
                if pa.types.is_list(col_type.value_type):
                    print(f"     - nested value type: {col_type.value_type.value_type}")
    
    # 保存合并后的数据
    pq.write_table(final_table, output_path / "data.parquet")
    
    # 更新并保存 metadata
    save_metadata()
    
    # 🔍 Validate saved file
    print(f"\n🔍 [Validation] Saved file validation:")
    saved_table = pq.read_table(output_path / "data.parquet")
    print(f"   - Read table column count: {len(saved_table.column_names)}")
    print(f"   - Read table row count: {len(saved_table)}")
    if 'text_sentences_sonar_emb' in saved_table.column_names:
        first_row_emb = saved_table['text_sentences_sonar_emb'][0]
        print(f"   - First row text_sentences_sonar_emb type: {type(first_row_emb)}")
        print(f"   - First row text_sentences_sonar_emb value type: {type(first_row_emb.as_py()) if hasattr(first_row_emb, 'as_py') else 'N/A'}")
        if hasattr(first_row_emb, 'as_py'):
            py_value = first_row_emb.as_py()
            print(f"   - Type after as_py(): {type(py_value)}")
            if isinstance(py_value, list) and len(py_value) > 0:
                print(f"   - as_py()[0] type: {type(py_value[0])}")
    
    print("\n" + "=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print(f"📁 输出文件: {output_dir}/data.parquet")
    print(f"📊 处理样本: {processed}")
    print(f"💾 文件大小: {(output_path / 'data.parquet').stat().st_size / 1024 / 1024:.2f} MB")
    print("\n📝 提示:")
    print("   多卡运行时，datacard 将在所有任务完成后统一更新")
    print("   如需手动更新，请运行:")
    print(f"   python scripts/update_datacards.py --output_dir={output_dir} --dataset_name={DATASET_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    import fire
    fire.Fire(prepare_fine_web)
