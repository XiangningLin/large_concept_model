# Dataloader Shuffle Seed 调查：LCM vs SentenceSSM

本文档汇总 LCM baselines 与 SentenceSSM 在 dataloader 层 shuffle 的 seed 可配置性。

---

## 1. LCM（large_concept_model）

### 1.1 机制

- **数据管线**：`lcm/datasets/dataloading.py` 使用 `DataLoadingConfig` 的 `shuffle` 和 `seed`
- **Shuffle 实现**：`parquet_utils.shuffle_table` + `np.random.RandomState(seed)`
- **Seed 来源**：`loading_config.seed`，经 `_get_inner_seed(rank, sharding_in_memory)` 计算

### 1.2 可配置性

| 参数 | 默认值 | 配置位置 | 说明 |
|------|--------|----------|------|
| `data_loading_config.seed` | `123` | `lcm/datasets/configs.py` | 可固定 shuffle seed |
| `data_loading_config.shuffle` | `True` | 同上 | 是否 shuffle |

### 1.3 固定 seed 的方式

**方式 A：Hydra 覆盖**

```bash
uv run torchrun ... -m lcm.train \
  ++trainer.data_loading_config.seed=42 \
  ...
```

**方式 B：在 recipe YAML 中设置**

```yaml
# recipes/train/pretrain/mse_60M.yaml
data_loading_config:
  batch_size: 32
  seed: 42   # 添加此行
```

### 1.4 分布式行为

- `world_size > 1` 时**必须**设置 `seed`（否则 assert）
- `sharding_in_memory=False`：`inner_seed = seed + rank * 100_000`（各 rank 不同）
- `sharding_in_memory=True`：`inner_seed = seed`（各 rank 相同，保证 sharding 一致）

---

## 2. SentenceSSM（MSE_trainer，TensorCache）

### 2.1 机制

- **数据集**：`TensorCacheIterableDataset`（`src/data_process/tensor_cache_iterable_dataset.py`）
- **Shuffle**：文件内 `rng.shuffle(indices)`，`rng = random.Random(seed)`
- **Seed 参数**：`seed: Optional[int] = 42`（默认 42）

### 2.2 可配置性

| 参数 | 默认值 | 配置位置 | 说明 |
|------|--------|----------|------|
| `seed` | `42` | `TensorCacheIterableDataset.__init__` | **无 config 传入** |

### 2.3 现状

`MSE_trainer.py` 创建数据集时**未传入 seed**：

```python
# MSE_trainer.py:381-386
self.dataset = TensorCacheIterableDataset(
    self.train_cache_dir, 
    shuffle=True,
    max_samples=self.train_max_samples,
    start_index=start_index
    # seed 未传入 → 使用默认 42
)
```

- 使用默认 `seed=42`，无法通过配置修改
- 有 RNG 状态保存/恢复（`set_rng_state`）用于 checkpoint resume，但不影响初始 seed

### 2.4 固定 seed 的可行改动

在 MSE_trainer 中增加 `dataset_seed` 参数，并传给 `TensorCacheIterableDataset`：

```python
# 需在 MSE_trainer 构造函数增加 dataset_seed 参数
# 并在创建 TensorCacheIterableDataset 时传入 seed=dataset_seed
```

同时需在调用 MSE_trainer 的入口（如 run script）增加对应 config 项。

---

## 3. SentenceSSM Baselines（Token Models，ProcessedIterableDataset）

### 3.1 机制

- **数据集**：`ProcessedIterableDataset`（`baselines/token_models/preprocessing/processed_iterable_dataset.py`）
- **Shuffle**：HuggingFace `dataset.shuffle(seed=..., buffer_size=...)`
- **Seed 参数**：`seed: Optional[int] = 42`

### 3.2 可配置性

| 参数 | 默认值 | 配置位置 | 说明 |
|------|--------|----------|------|
| `shuffle_buffer_size` | 来自 config | `config.task.shuffle_buffer_size` | ✅ 可配置 |
| `seed` | `42` | `TokenModelTrainer` 硬编码 | ❌ 不可配置 |

### 3.3 现状

`TokenModelTrainer` 创建数据集时**硬编码 seed=42**：

```python
# baselines/token_models/training/trainer.py:106-122
self.datasets = {
    "train": ProcessedIterableDataset(
        data_path=data_paths[0],
        streaming=streaming,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=42   # 硬编码
    ),
    ...
}
```

- `shuffle_buffer_size` 来自 `config.task.shuffle_buffer_size`（如 `task/train.yaml` 中 10000）
- `seed` 无 config 项，无法修改

### 3.4 固定 seed 的可行改动

1. 在 `TaskConfig` schema 中增加 `shuffle_seed: int = 42`
2. 在 `run_training.py` 中读取 `config.task.shuffle_seed` 并传给 `TokenModelTrainer`
3. 在 `TokenModelTrainer` 中用该参数替代硬编码的 `seed=42`

---

## 4. 对比汇总

| 项目 | Shuffle 位置 | Seed 可配置 | 固定 seed 方式 |
|------|--------------|-------------|----------------|
| **LCM** | `parquet_utils` + `DataLoadingConfig` | ✅ 是 | `++trainer.data_loading_config.seed=42` 或在 recipe 中设置 |
| **SentenceSSM MSE** | `TensorCacheIterableDataset` 文件内 | ❌ 否 | 需改 MSE_trainer 增加 `dataset_seed` 并传入 |
| **SentenceSSM Baselines** | `ProcessedIterableDataset.shuffle()` | ❌ 否 | 需改 schema + run_training + trainer 增加 `shuffle_seed` |

---

## 5. 建议

1. **LCM**：已支持，直接使用 `++trainer.data_loading_config.seed=<value>` 即可。
2. **SentenceSSM MSE**：建议在 MSE_trainer 及调用链中增加 `dataset_seed` 参数，并传入 `TensorCacheIterableDataset(..., seed=dataset_seed)`。
3. **SentenceSSM Baselines**：建议在 task config 中增加 `shuffle_seed`，并在 trainer 中替换硬编码的 `seed=42`。
