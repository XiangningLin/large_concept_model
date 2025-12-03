# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Streaming fine web Data Preparation Script
真正的流式下载和处理，边下载边处理，不需要下载完整数据集
"""

import os
os.environ["HF_HOME"] = "/projects/p32721/large_concept_model/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/projects/p32721/large_concept_model/hf_cache/datasets"

from pathlib import Path
import sys
import torch
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sentence_splitter import SentenceSplitter
from stopes.utils.arrow_utils import nested_numpy_to_pyarrow


def prepare_fine_web(
    output_dir: str = "output/fine_web",
    num_samples: int = 100,
    start_index: int = 0,
    batch_size: int = 10,
    max_sentence_length: int = 256,
    add_split_column: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """
    流式处理fine web数据集
    
    参数:
        output_dir: 输出目录
        num_samples: 要处理的样本数量
        start_index: 起始索引（用于多GPU并行）
        batch_size: 批处理大小（用于SONAR编码）
        max_sentence_length: 最大句子长度
        add_split_column: 是否添加 split 列用于 train/validation 分区（默认: True）
        train_ratio: 训练数据比例，当 add_split_column=True 时使用（默认: 0.8，即 80% 训练，20% 验证）
        seed: 随机种子，用于 split 的可重现性（默认: 42）
    
    使用示例:
        # 100条样本（5-10分钟）
        python scripts/prepare_fine_web.py --num_samples=100
        
        # 1000条样本（30-60分钟）
        python scripts/prepare_fine_web.py --num_samples=1000 --output_dir=output/fine_web_1k
        
        # 10000条样本（4-6小时）
        python scripts/prepare_fine_web.py --num_samples=10000 --output_dir=output/fine_web_10k

        错误报告：
        我们遇到的主要错误是pyarrow在读取我们的parquet文件时，由于LCM的训练approach硬性要求sonar embedding为
        list<fixed_size_list[1024]>（即固定长度列表）格式，而我们的是可变长度列表。
        我们通过部署stopes的nested_numpy_to_pyarrow函数将可变长度列表转换为固定长度列表，从而解决了这个问题。
        见第270行代码：text_sentences_sonar_emb_pa = nested_numpy_to_pyarrow(all_embeddings)。
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 fine web 流式数据处理")
    print("=" * 80)
    print(f"📁 输出目录: {output_dir}")
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
    
    # 使用tqdm显示进度
    with tqdm(total=num_samples, desc="处理进度") as pbar:
        batch_texts = []
        batch_originals = []
        
        for idx, sample in enumerate(dataset):
            # 跳过start_index之前的样本
            if idx < start_index:
                continue
            if processed >= num_samples:
                break
            
            text = sample['text'].strip()
            if not text:
                continue
            
            # 分句
            sentences = splitter.split(text)
            if not sentences:
                continue
            
            # 截断过长的句子
            sentences = [s[:max_sentence_length] for s in sentences]
            
            batch_texts.append(sentences)
            batch_originals.append({
                'url': sample.get('url', ''),
                'timestamp': sample.get('timestamp', ''),
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
                        
                        # 保存数据
                        all_data.append({
                            'text_sentences': sents,
                            'text_sentences_sonar_emb': embeddings.cpu().numpy(),
                            'url': batch_originals[i]['url'],
                            'timestamp': batch_originals[i]['timestamp'],
                        })
                        
                        processed += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n⚠️  Processing failed: {str(e)}")
                        continue
                
                batch_texts = []
                batch_originals = []
        
        # 处理剩余的数据
        if batch_texts and processed < num_samples:
            for i, sents in enumerate(batch_texts):
                if processed >= num_samples:
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
                    
                    all_data.append({
                        'text_sentences': sents,
                        'text_sentences_sonar_emb': embeddings.cpu().numpy(),
                        'url': batch_originals[i]['url'],
                        'timestamp': batch_originals[i]['timestamp'],
                    })
                    
                    processed += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n⚠️  Processing failed: {str(e)}")
                    continue
    
    # 添加 split 列（如果启用）
    if add_split_column:
        import random
        random.seed(seed)
        
        total_samples = len(all_data)
        train_size = int(total_samples * train_ratio)
        
        # 随机打乱索引
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = set(indices[:train_size])
        
        for i in range(total_samples):
            if i in train_indices:
                all_data[i]['split'] = 'train'
            else:
                all_data[i]['split'] = 'validation'
        
        print(f"\n✅ Added split column: {train_size} train, {total_samples - train_size} validation")
    
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
    
    table = pa.table({
        'text_sentences': text_sentences_list,
        'text_sentences_sonar_emb': text_sentences_sonar_emb_pa,
        'url': [d['url'] for d in all_data],
        'timestamp': [d['timestamp'] for d in all_data],
        'split': [d.get('split', 'train') for d in all_data],  # Add split column, default to 'train'
    })
    
    # 🔍 Validate PyArrow table schema
    print(f"\n🔍 [Validation] PyArrow Table Schema:")
    print(f"   - Table column count: {len(table.column_names)}")
    print(f"   - Table row count: {len(table)}")
    for col_name in table.column_names:
        col_type = table[col_name].type
        print(f"   - {col_name}: {col_type}")
        if col_name == 'text_sentences_sonar_emb':
            print(f"     - is list type: {pa.types.is_list(col_type)}")
            if pa.types.is_list(col_type):
                print(f"     - value type: {col_type.value_type}")
                if pa.types.is_list(col_type.value_type):
                    print(f"     - nested value type: {col_type.value_type.value_type}")
    
    pq.write_table(table, output_path / "data.parquet")
    
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
    print("\n📝 下一步:")
    print("1. 更新 lcm/datacards/datacards.yaml:")
    print(f"   parquet_path:")
    print(f"     local: \"{output_dir}\"")
    print("\n2. 开始训练:")
    print("   bash start_pretrain_780m.sh")
    print("=" * 80)


if __name__ == "__main__":
    import fire
    fire.Fire(prepare_fine_web)
