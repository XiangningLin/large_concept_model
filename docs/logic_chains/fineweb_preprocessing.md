# FineWeb 数据预处理逻辑链分析

本文档描述 LCM 中 **FineWeb** 数据预处理的完整逻辑链：从入口脚本到输出 Parquet 与 Datacard 的更新。

---

## 1. 入口与调用关系

```
prepare_data.sh                    (多 GPU 并行入口)
    │
    ├─→ 对每个 GPU i: 后台启动
    │       .venv/bin/python -u scripts/prepare_fine_web.py
    │           --output_dir=$OUTPUT_DIR/shard_$i
    │           --num_samples=$NUM_SAMPLES_ARG
    │           --start_index=$START_IDX
    │           --batch_size=$BATCH_SIZE
    │           [--max_sentence_length=...] 等
    │       输出: logs/prepare_gpu${i}.log
    │
    └─→ wait 所有任务完成后
            .venv/bin/python scripts/update_datacards.py
                --output_dir="$OUTPUT_DIR"   # 不含 shard，统一写 datacard
                --dataset_name=fine_web_edu
                ...
```

- **单机单卡 / 调试**：可直接运行  
  `python scripts/prepare_fine_web.py --output_dir=... --num_samples=100`（入口为 `prepare_fine_web()`，经 `fire.Fire(prepare_fine_web)` 解析 CLI）。
- **多卡并行**：由 `prepare_data.sh` 按 `start_index` 分片，每个 GPU 写 `output_dir/shard_$i/`，最后**只调用一次** `update_datacards.py`，`--output_dir` 指向不含 `shard_*` 的顶层目录（datacard 中 parquet_path 指向该顶层目录，训练时由 dataloader 按配置读取各 shard）。

---

## 2. 环境与常量

| 位置 | 作用 |
|------|------|
| 脚本开头 | 若未设置则默认 `HF_HOME`、`HF_DATASETS_CACHE`，保证 HuggingFace 缓存目录 |
| `DATASET_NAME = "fine_web_edu"` | 与 `update_datacards.py` 使用的 dataset_name 一致 |
| `CLUSTER_NAME = "s3"` | datacard 中 parquet_path 的 key（与脚本内逻辑一致，实际可为本地路径） |

---

## 3. 主流程概览（prepare_fine_web）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 参数与输出路径                                                        │
│    num_samples / start_index / batch_size / max_sentence_length /        │
│    add_split_column / train_ratio / seed / checkpoint_interval            │
│    → output_dir, output_file=data.parquet, metadata_file, checkpoint_dir │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. 断点恢复（二选一或顺序）                                               │
│    A) 检查点恢复：checkpoint_dir 下 data_checkpoint_*.parquet            │
│       → 合并为 existing_data，resume_from_index = metadata 或估算        │
│    B) 无检查点时：若存在 output_file → 读入 existing_data，判断是否      │
│       可恢复（样本数/ metadata last_processed_index）→ resume_from_index  │
│    若已完整则直接 return；否则用 existing_data + resume_from_index 继续   │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. 设备与模型初始化                                                      │
│    SentenceSplitter(language='en')                                      │
│    TextToEmbeddingModelPipeline(encoder/tokenizer="text_sonar_basic_...") │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. 流式数据源                                                            │
│    load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",              │
│                 split="train", streaming=True)                           │
│    → 按索引迭代，不一次性下载全量                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. 主循环（按 dataset 索引）                                             │
│    for idx, sample in enumerate(dataset):                                │
│      - 若 idx < start_index → continue                                  │
│      - 若非 process_all 且 processed >= num_samples → break              │
│      - text = sample['text'].strip()；空则 continue                      │
│      - sentences = splitter.split(text)；空则 continue                   │
│      - sentences = [s[:max_sentence_length] for s in sentences]         │
│      - 累积 batch_texts / batch_originals                               │
│      - 当 len(batch_texts) >= batch_size:                               │
│          for 每篇: sonar_pipeline.predict(sents, source_lang="eng_Latn") │
│          → 得到 embeddings (numpy)，拼成 sample_data                     │
│          → 若启用检查点：写入 current_checkpoint_data，满 checkpoint_interval│
│            则 save_checkpoint() 并写 metadata；否则 append all_data       │
│          batch_texts/originals 清空                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. 剩余 batch 处理                                                       │
│    与上面相同逻辑处理最后不足 batch_size 的 batch_texts                  │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. 检查点与合并                                                          │
│    - 若启用检查点且 current_checkpoint_data 非空：再 save_checkpoint 一次│
│    - 若启用检查点：从 checkpoint_dir 所有 data_checkpoint_*.parquet      │
│      合并为 new_table（含 split 列已在 save_checkpoint 时写入）           │
│    - 否则：用 all_data 打 split（train_ratio/seed），再                   │
│      nested_numpy_to_pyarrow(all_embeddings) → 建 new_table              │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. 与旧数据合并并写盘                                                    │
│    - 若有 existing_data 且未用检查点合并：final_table = concat(existing,  │
│      new_table)，schema 对齐                                             │
│    - 否则 final_table = new_table                                       │
│    - pq.write_table(final_table, output_path / "data.parquet")           │
│    - save_metadata()                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键数据流与结构

- **单条样本（内存）**  
  - 输入：`sample['text']`，以及可选的 `url`、`timestamp`。  
  - 中间：`sentences`（字符串列表）、`embeddings`（SONAR 编码后的 numpy，形状与句子数×dim 对应）。  
  - 写入前：  
    - `text_sentences`: 字符串列表（可能截断到 `max_sentence_length`）。  
    - `text_sentences_sonar_emb`: 先为 numpy，再经 `nested_numpy_to_pyarrow` 转为 PyArrow 的 **list&lt;fixed_size_list[1024]&gt;**（满足 LCM 训练对固定长度 list 的要求）。  
    - `url`、`timestamp`、`split`（train/validation）。

- **Embedding 格式**  
  - SONAR 输出为每句一个向量（如 1024 维）。  
  - 每篇文档对应一个「句子数×1024」的数组，在 batch 里是 list of arrays。  
  - 写入 Parquet 前必须用 `stopes.utils.arrow_utils.nested_numpy_to_pyarrow(all_embeddings)` 转为固定长度列表，否则训练端读取会报错（见脚本注释）。

- **Split 列**  
  - 若 `add_split_column=True`：  
    - 检查点模式：在 `save_checkpoint()` 里按 `(global_idx, seed)` 的哈希与 `train_ratio` 写 `split`。  
    - 非检查点模式：在合并前对「新数据」用 `total_samples` 与 `train_ratio`/`seed` 分配 train/validation。  
  - 保证同一全局索引在不同运行/恢复下 split 一致（依赖 seed 与全局索引）。

---

## 5. 断点恢复逻辑小结

| 情况 | 行为 |
|------|------|
| 存在 `checkpoint_dir/data_checkpoint_*.parquet` | 合并为 `existing_data`，从 metadata 或 `start_index + existing_samples` 得到 `resume_from_index`，并更新 `next_checkpoint_number`。 |
| 无检查点但存在 `output_file` | 读取为 `existing_table`，根据 `process_all`/`num_samples` 与 metadata 的 `last_processed_index` 判断是否已完整；若可恢复则 `resume_from_index` 继续，否则删文件重跑。 |
| 已完整（样本数达标或 process_all 且文件合理） | 直接 `return`，不再处理。 |
| 恢复后 | `start_index := resume_from_index`；若非 process_all 则 `num_samples := remaining_samples`；后续循环从 `idx < start_index` 跳过，只处理新样本，最后与 `existing_data` 合并写回。 |

`save_metadata()` 在正常结束和 SIGINT/SIGTERM 时都会被调用（atexit + signal handler），写入 `last_processed_index`、`processed_samples`、`checkpoint_count` 等，供下次恢复使用。

---

## 6. 输出与下游

- **单 shard 输出（prepare_fine_web 单次运行）**  
  - `output_dir/data.parquet`：最终表，列包括 `text_sentences`、`text_sentences_sonar_emb`、`url`、`timestamp`、`split`。  
  - `output_dir/.progress_metadata.json`：进度与恢复用。  
  - 若启用检查点：`output_dir/checkpoints/data_checkpoint_*.parquet`；完成后会合并进 `data.parquet`（逻辑上可视为中间产物）。

- **多 shard 时（prepare_data.sh）**  
  - 每个 GPU 写 `$OUTPUT_DIR/shard_$i/data.parquet`。  
  - `update_datacards.py` 只调用一次，`--output_dir="$OUTPUT_DIR"`（不含 shard），这样 datacard 里的 `parquet_path` 指向的是「包含多个 shard 子目录」的根目录；训练时由 LCM 的 dataloader 根据 datacard 和配置去扫描这些 shard（具体见 `lcm/datasets` 的配置与 sharding 逻辑）。

- **Datacard 更新（update_datacards.py）**  
  - 根据 `generate_datacard_entry()` 生成 YAML 片段：`name`、`parquet_path`（local/s3）、`source_column`、`source_text_column`，以及可选的 partition 注释（split、train_ratio、seed）。  
  - 写回 `lcm/datacards/datacards.yaml`（更新已存在 name 或追加新文档）。

---

## 7. 依赖与关键引用

| 依赖 | 用途 |
|------|------|
| `datasets.load_dataset(..., streaming=True)` | 流式加载 FineWeb，按索引迭代 |
| `sentence_splitter.SentenceSplitter` | 英文分句 |
| `sonar.inference_pipelines.text.TextToEmbeddingModelPipeline` | 文本 → SONAR 向量 |
| `stopes.utils.arrow_utils.nested_numpy_to_pyarrow` | 可变长度 list of arrays → list&lt;fixed_size_list[1024]&gt;，满足训练 schema |
| `pyarrow` (pq, pa.table) | 写 Parquet、合并表、schema 校验 |
| `fire` | CLI 解析，映射到 `prepare_fine_web(...)` 参数 |

---

## 8. 文件清单（逻辑链涉及）

| 文件 | 角色 |
|------|------|
| `prepare_data.sh` | 多 GPU 调度、调用 prepare_fine_web.py 与 update_datacards.py |
| `scripts/prepare_fine_web.py` | FineWeb 流式下载、分句、SONAR 编码、检查点、合并、写 Parquet 与 metadata |
| `scripts/update_datacards.py` | 生成 datacard 片段并写回 `lcm/datacards/datacards.yaml` |

以上为 FineWeb 数据预处理的完整逻辑链；其他数据源（如 C4、Wikipedia、ToM）有各自的 prepare_* 脚本与入口，可参考本结构单独写逻辑链文档。
