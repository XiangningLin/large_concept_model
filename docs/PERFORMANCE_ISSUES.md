# 数据预处理性能问题分析

## 🔴 严重问题：批处理未充分利用 GPU

### 问题位置
- `scripts/prepare_fine_web.py` 第 476-540 行
- `scripts/prepare_c4.py` 第 135-158 行

### 问题描述
虽然代码收集了 `batch_size` 个样本到 `batch_texts`，但在处理时**逐个**调用 `sonar_pipeline.predict()`，而不是批量处理。

### 当前代码（错误示例）
```python
# 批量编码
if len(batch_texts) >= batch_size:
    # 处理这一批
    for i, sents in enumerate(batch_texts):  # ❌ 逐个处理
        try:
            # 编码句子
            embeddings = sonar_pipeline.predict(  # ❌ 每个样本单独调用GPU
                sents,
                source_lang="eng_Latn"
            )
```

### 性能影响
1. **GPU 利用率极低**：每个样本单独调用 GPU，无法充分利用 GPU 的并行能力
2. **频繁的 GPU kernel 启动开销**：每次调用都有启动开销，累积起来非常耗时
3. **批处理优势完全丢失**：虽然收集了 batch，但没有利用批处理的并行优势
4. **预计性能损失**：可能只有 10-20% 的 GPU 利用率，速度可能慢 5-10 倍

### 应该的优化方式
需要检查 SONAR pipeline 是否支持批量处理多个文档。如果支持，应该：
```python
# 批量编码所有文档
if len(batch_texts) >= batch_size:
    # 一次性处理整个 batch
    all_embeddings = sonar_pipeline.predict_batch(  # 批量处理
        batch_texts,
        source_lang="eng_Latn"
    )
    # 然后逐个保存结果
    for i, embeddings in enumerate(all_embeddings):
        # 保存数据...
```

---

## 🟡 中等问题：数据流串行处理

### 问题位置
- `scripts/prepare_fine_web.py` 第 446-603 行
- `scripts/prepare_c4.py` 第 109-186 行

### 问题描述
虽然 `prepare_data.sh` 启动了多个 GPU 进程并行处理，但每个进程内部是**完全串行**的：
1. 从数据集读取一个样本
2. 分句
3. 等待 GPU 编码（由于上面的问题，GPU 利用率低）
4. 保存到内存
5. 重复

### 性能影响
- 数据加载和 GPU 计算没有重叠（pipeline parallelism）
- GPU 在等待数据加载时处于空闲状态
- 预计可以提升 20-30% 的性能

### 优化建议
使用异步处理或多线程：
```python
import queue
import threading

# 数据加载线程
def data_loader_thread(dataset, data_queue):
    for sample in dataset:
        # 预处理
        sentences = splitter.split(sample['text'])
        data_queue.put((sentences, metadata))

# GPU 处理线程
def gpu_process_thread(sonar_pipeline, data_queue, result_queue):
    batch = []
    while True:
        item = data_queue.get()
        batch.append(item)
        if len(batch) >= batch_size:
            # 批量处理
            embeddings = sonar_pipeline.predict_batch([x[0] for x in batch])
            result_queue.put(embeddings)
            batch = []
```

---

## 🟡 中等问题：内存累积

### 问题位置
- `scripts/prepare_fine_web.py` 第 337 行：`all_data = []`
- `scripts/prepare_c4.py` 第 101 行：`all_data = []`

### 问题描述
所有处理后的数据都累积在 `all_data` 列表中，直到最后才写入文件。

### 性能影响
- 对于大数据集，可能导致内存不足
- 虽然启用了检查点机制，但默认情况下还是会累积数据
- 可能导致 OOM（Out of Memory）错误

### 优化建议
1. 更频繁地写入检查点（减小 `checkpoint_interval`）
2. 使用流式写入，而不是累积后写入
3. 对于大数据集，强制启用检查点

---

## 🟢 轻微问题：默认 batch_size 可能偏小

### 问题位置
- `prepare_data.sh` 第 14 行：`BATCH_SIZE=20`
- `scripts/prepare_fine_web.py` 第 39 行：`batch_size: int = 10`

### 问题描述
默认的 `batch_size=10` 或 `20` 可能对于现代 GPU（如 A100）来说偏小。

### 性能影响
- 较小的 batch size 无法充分利用 GPU 的并行能力
- 对于 A100 等大 GPU，可以尝试更大的 batch size（如 32, 64, 128）

### 优化建议
根据 GPU 型号调整默认 batch_size：
- A100: 64-128
- V100: 32-64
- 较小 GPU: 16-32

---

## 📊 性能影响总结

| 问题 | 严重程度 | 预计性能损失 | 修复难度 |
|------|---------|------------|---------|
| 批处理未充分利用 GPU | 🔴 严重 | 5-10x 慢 | 中等 |
| 数据流串行处理 | 🟡 中等 | 20-30% 慢 | 较难 |
| 内存累积 | 🟡 中等 | 可能导致 OOM | 容易 |
| batch_size 偏小 | 🟢 轻微 | 10-20% 慢 | 容易 |

---

## 🚀 优化优先级

1. **最高优先级**：修复批处理问题（预计可提升 5-10x 性能）
2. **高优先级**：增加 batch_size（预计可提升 10-20% 性能）
3. **中优先级**：实现异步数据加载（预计可提升 20-30% 性能）
4. **低优先级**：优化内存使用（防止 OOM）

---

## 🔍 需要进一步调查

1. **SONAR pipeline 的批处理能力**：
   - 检查 `TextToEmbeddingModelPipeline` 是否支持批量处理多个文档
   - 查看 SONAR 文档或源码，确认批处理 API

2. **GPU 利用率监控**：
   - 运行 `nvidia-smi` 监控 GPU 利用率
   - 如果利用率 < 50%，说明存在严重的性能问题

3. **数据加载瓶颈**：
   - 检查 HuggingFace streaming 数据加载速度
   - 确认网络带宽是否成为瓶颈

