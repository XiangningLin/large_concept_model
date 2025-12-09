# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Streaming C4 Data Preparation Script
真正的流式下载和处理，边下载边处理，不需要下载完整数据集
"""

import os
# 使用大空间目录存储 Hugging Face 缓存（如果环境变量未设置则使用此默认值）
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/work/hdd/bfaq/jlyu3/lcm/hf_cache"
if "HF_DATASETS_CACHE" not in os.environ:
    os.environ["HF_DATASETS_CACHE"] = "/work/hdd/bfaq/jlyu3/lcm/hf_cache/datasets"

from pathlib import Path
import torch
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sentence_splitter import SentenceSplitter


def prepare_c4_streaming(
    output_dir: str = "output/c4_streaming",
    num_samples: int = 100,
    start_index: int = 0,
    batch_size: int = 10,
    max_sentence_length: int = 256,
):
    """
    流式处理C4数据集
    
    参数:
        output_dir: 输出目录
        num_samples: 要处理的样本数量
        start_index: 起始索引（用于多GPU并行）
        batch_size: 批处理大小（用于SONAR编码）
        max_sentence_length: 最大句子长度
    
    使用示例:
        # 100条样本（5-10分钟）
        python scripts/prepare_c4.py --num_samples=100
        
        # 1000条样本（30-60分钟）
        python scripts/prepare_c4.py --num_samples=1000 --output_dir=output/c4_1k
        
        # 10000条样本（4-6小时）
        python scripts/prepare_c4.py --num_samples=10000 --output_dir=output/c4_10k
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 C4 流式数据处理")
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
    
    # 流式加载C4数据集
    print(f"📥 开始流式下载C4数据 (前 {num_samples} 条)...")
    print("   (这是真正的流式，只下载需要的数据)\n")
    
    dataset = load_dataset(
        "allenai/c4",
        "en",
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
                        print(f"\n⚠️  处理失败: {str(e)}")
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
                    
                    all_data.append({
                        'text_sentences': sents,
                        'text_sentences_sonar_emb': embeddings.cpu().numpy(),
                        'url': batch_originals[i]['url'],
                        'timestamp': batch_originals[i]['timestamp'],
                    })
                    
                    processed += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n⚠️  处理失败: {str(e)}")
                    continue
    
    # 保存为Parquet文件
    print(f"\n💾 保存数据到 {output_dir}/data.parquet ...")
    
    # 将嵌入转换为列表格式（PyArrow兼容）
    table = pa.table({
        'text_sentences': [d['text_sentences'] for d in all_data],
        'text_sentences_sonar_emb': [d['text_sentences_sonar_emb'].tolist() for d in all_data],
        'url': [d['url'] for d in all_data],
        'timestamp': [d['timestamp'] for d in all_data],
    })
    
    pq.write_table(table, output_path / "data.parquet")
    
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
    fire.Fire(prepare_c4_streaming)
