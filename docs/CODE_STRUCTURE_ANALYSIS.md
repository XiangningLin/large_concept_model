# prepare_fine_web.py 代码结构分析

## 📋 核心逻辑（Essential Logic）

这些是数据处理的**核心流程**，去掉这些功能就无法正常工作：

### 1. 数据读取和预处理循环（第 446-474 行）
```python
for idx, sample in enumerate(dataset):
    # 跳过start_index之前的样本
    if idx < start_index:
        continue
    
    text = sample['text'].strip()
    sentences = splitter.split(text)
    sentences = [s[:max_sentence_length] for s in sentences]
    
    batch_texts.append(sentences)
    batch_originals.append({...})
```
**功能**：从数据集读取样本，分句，累积到 batch

---

### 2. 核心批处理逻辑（第 476-484 行）
```python
if len(batch_texts) >= batch_size:
    all_embeddings = sonar_pipeline.predict(
        batch_texts,  # 传入整个 batch: List[List[str]]
        source_lang="eng_Latn"
    )
```
**功能**：**这是最核心的部分** - 批量调用 SONAR pipeline，一次性处理整个 batch

---

### 3. 保存处理结果（第 520-538 行）
```python
for i, (sents, embeddings) in enumerate(zip(batch_texts, embeddings_list)):
    emb_numpy = embeddings.cpu().numpy()  # 转换为 numpy
    sample_data = {
        'text_sentences': sents,
        'text_sentences_sonar_emb': emb_numpy,
        'url': batch_originals[i]['url'],
        'timestamp': batch_originals[i]['timestamp'],
    }
    all_data.append(sample_data)
    processed += 1
```
**功能**：将批处理结果转换为 numpy 并保存到内存

---

### 4. 处理剩余数据（第 566-638 行）
```python
if batch_texts and (process_all or processed < num_samples):
    all_embeddings = sonar_pipeline.predict(batch_texts, ...)
    # 保存结果...
```
**功能**：循环结束后，处理未达到 batch_size 的剩余样本

---

### 5. 最终保存到文件（第 682-763 行）
```python
text_sentences_sonar_emb_pa = nested_numpy_to_pyarrow(all_embeddings)
final_table = pa.table({...})
pq.write_table(final_table, output_path / "data.parquet")
```
**功能**：将所有数据转换为 PyArrow 格式并保存为 parquet 文件

---

## 🔧 附加逻辑（Supporting/Helper Logic）

这些是**增强功能**，用于健壮性、调试、恢复等，但不是核心流程：

### A. 断点恢复功能（第 110-284 行）
**功能**：检查已存在的检查点文件或输出文件，尝试从断点继续
- 检查检查点文件（第 118-172 行）
- 检查输出文件（第 175-284 行）
- 计算恢复索引
- **可以简化**：如果不需要断点恢复，可以删除

---

### B. 检查点功能（第 102-108, 351-394, 544-557 行）
**功能**：定期保存中间结果，防止数据丢失
- `save_checkpoint()` 函数（第 352-394 行）
- 检查点间隔检查（第 549-554 行）
- **可以简化**：如果不需要检查点，可以禁用（`checkpoint_interval=0`）

---

### C. Metadata 保存（第 396-416, 560-561, 637-638 行）
**功能**：保存处理进度，用于断点恢复
- `save_metadata()` 函数（第 397-416 行）
- 每 100 个样本保存一次（第 560-561 行）
- **可以简化**：如果不需要断点恢复，可以删除

---

### D. 信号处理（第 421-436 行）
**功能**：处理 Ctrl+C 等中断信号，保存进度
```python
def signal_handler(signum, frame):
    save_checkpoint(...)
    save_metadata()
    sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)
```
**可以简化**：如果不需要优雅退出，可以删除

---

### E. 数据验证和调试信息（第 486-495 行）
**功能**：打印第一个 batch 的 embeddings 信息，用于调试
```python
if processed == 0:
    print(f"🔍 [Validation] Batch embeddings info:")
    print(f"   - all_embeddings type: {type(all_embeddings)}")
    ...
```
**可以简化**：调试完成后可以删除或改为可选

---

### F. 返回格式处理（第 500-510 行）
**功能**：处理 SONAR pipeline 可能返回的不同格式
```python
if isinstance(all_embeddings, (list, tuple)):
    embeddings_list = all_embeddings
elif hasattr(all_embeddings, 'shape') and len(all_embeddings.shape) > 1:
    embeddings_list = ...
else:
    embeddings_list = [all_embeddings]
```
**可以简化**：如果确定 SONAR 总是返回 list，可以简化

---

### G. 数据格式转换（第 522-527, 606-611 行）
**功能**：处理不同格式的 embeddings（torch tensor / numpy / 其他）
```python
if hasattr(embeddings, 'cpu'):
    emb_numpy = embeddings.cpu().numpy()
elif hasattr(embeddings, 'numpy'):
    emb_numpy = embeddings.numpy()
else:
    emb_numpy = np.array(embeddings)
```
**可以简化**：如果确定 SONAR 总是返回 torch tensor，可以简化为 `embeddings.cpu().numpy()`

---

### H. 空数据过滤（第 454-463 行）
**功能**：跳过空文本或空句子
```python
if not text:
    continue
if not sentences:
    continue
```
**核心逻辑**：这个应该保留，避免处理无效数据

---

### I. 样本数量限制检查（第 450-452, 570-576, 602-603 行）
**功能**：检查是否达到目标样本数
```python
if not process_all and processed >= num_samples:
    break
```
**核心逻辑**：这个应该保留，用于控制处理数量

---

### J. Split 列分配（第 359-373, 649-680 行）
**功能**：为数据分配 train/validation split
```python
if add_split_column:
    if hash((global_idx, seed)) % 100 < int(train_ratio * 100):
        d['split'] = 'train'
    else:
        d['split'] = 'validation'
```
**可以简化**：如果不需要 split，可以禁用（`add_split_column=False`）

---

### K. 进度条显示（第 442 行）
**功能**：使用 tqdm 显示处理进度
```python
with tqdm(total=total_expected, desc="处理进度", unit="样本", initial=existing_samples) as pbar:
    pbar.update(1)
```
**可以简化**：如果不需要进度条，可以删除

---

## 📊 代码复杂度分析

### 核心逻辑代码量
- **约 50-80 行**（数据读取、批处理、保存）

### 附加逻辑代码量
- **约 600-700 行**（断点恢复、检查点、metadata、验证等）

### 比例
- **核心逻辑：~10%**
- **附加逻辑：~90%**

---

## 🎯 简化建议

如果要简化代码，可以：

1. **保留核心逻辑**：
   - 数据读取循环
   - 批处理调用 `sonar_pipeline.predict(batch_texts)`
   - 结果保存
   - 剩余数据处理

2. **可以删除的附加功能**（如果不需要）：
   - 断点恢复（第 110-284 行）
   - 检查点功能（如果 `checkpoint_interval=0`）
   - Metadata 保存（如果不需要断点恢复）
   - 信号处理（如果不需要优雅退出）
   - 调试信息打印（第 486-495 行）

3. **可以简化的部分**：
   - 返回格式处理（如果确定 SONAR 返回格式）
   - 数据格式转换（如果确定返回类型）

---

## 🔑 最简化的核心代码结构

如果只保留核心逻辑，代码大概是这样的：

```python
# 初始化
sonar_pipeline = TextToEmbeddingModelPipeline(...)
splitter = SentenceSplitter(language='en')
batch_texts = []
batch_originals = []
all_data = []

# 核心循环
for idx, sample in enumerate(dataset):
    if idx < start_index:
        continue
    if processed >= num_samples:
        break
    
    sentences = splitter.split(sample['text'])
    batch_texts.append(sentences)
    batch_originals.append({...})
    
    # 核心批处理
    if len(batch_texts) >= batch_size:
        all_embeddings = sonar_pipeline.predict(batch_texts, source_lang="eng_Latn")
        
        for i, (sents, embeddings) in enumerate(zip(batch_texts, all_embeddings)):
            all_data.append({
                'text_sentences': sents,
                'text_sentences_sonar_emb': embeddings.cpu().numpy(),
                ...
            })
        batch_texts = []
        batch_originals = []

# 处理剩余数据
if batch_texts:
    all_embeddings = sonar_pipeline.predict(batch_texts, source_lang="eng_Latn")
    # 保存结果...

# 保存文件
final_table = pa.table({...})
pq.write_table(final_table, output_path / "data.parquet")
```

**约 30-40 行核心代码** vs **当前 834 行**

