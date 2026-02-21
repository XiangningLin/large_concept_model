# LCM 与我们的代码（SentenceSSM src + baselines）能力对比

本文档对比 **LCM 官方实现** 与 **我们的代码**（SentenceSSM 的 `src/` 与 `baselines/token_models/`）在**预处理**和**训练**端能完成的“动作”，并标出 LCM 目前**无法做到**的能力。

---

## 一、预处理端对比


| 能力                            | 我们的代码                                                                                                                                                     | LCM                                                                               | LCM 是否支持                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------- |
| **智能引号等 Unicode 标点 → ASCII**  | ✅ `normalize_unicode_punctuation`（smart quotes、em dash、angle quotes 等 → 英文引号/ASCII） 位置：`src/utils/preprocess_filter_utils.py`，在 `TrajectoryProcessor` 中调用 | ✅ `scripts/utils/preprocess_filter_utils.py` + `prepare_fine_web.py` 分句前调用        | ✅ **已支持**                 |
| **start_index（分片并行）**         | ✅ `task.start_index`，用于多进程/多卡分片 src：`prepare_data.py`、`data_processing_pipeline_parquet_acc.py` baselines：`data_processing_pipeline_token_parquet_acc.py` | ✅ `start_index` 参数，用于多 GPU 并行 `prepare_fine_web.py`，`prepare_data.sh` 按 GPU 分配    | ✅ 支持                      |
| **max_samples / 样本数限制**       | ✅ `task.max_samples`，从 start_index 起限制样本数                                                                                                                 | ✅ `num_samples`，支持指定数量或 "all"                                                     | ✅ 支持                      |
| **流式加载**                      | ✅ HuggingFace `streaming=True` 或 IterableDataset                                                                                                          | ✅ `load_dataset(..., streaming=True)`                                             | ✅ 支持                      |
| **断点恢复**                      | ✅ 基于 metadata / 已写 parquet 的进度恢复                                                                                                                          | ✅ 基于 checkpoint 文件与 `.progress_metadata.json`                                     | ✅ 支持                      |
| **train/val 划分**              | ✅ `auto_split` + hash 划分，或预定义 split                                                                                                                       | ✅ `add_split_column` + `train_ratio` + seed                                       | ✅ 支持                      |
| **UNK 字符过滤/替换**               | ✅ `filter_remaining_unk_chars`、`drop_text_with_unk_tokens`（src）                                                                                           | ✅ `drop_text_with_unk_tokens`，`prepare_fine_web.py` 分句后过滤，`enable_unk_filter` 可关闭 | ✅ **已支持**                 |
| **Tensor 缓存**（sentence-level） | ✅ src 支持 tensor cache，避免重复 embedding                                                                                                                      | ❌ 无，直接写 Parquet                                                                   | ❌ **不支持**                 |
| **多 tokenizer 支持**            | ✅ baselines 支持 Mamba/Qwen 等                                                                                                                               | ❌ 仅 SONAR 编码                                                                      | N/A（LCM 为 sentence-level） |


---

## 二、训练端对比


| 能力                         | 我们的代码                                                                                                                                                                                           | LCM                                                               | LCM 是否支持  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------- |
| **Token 里程碑 checkpoint**   | ✅ `checkpoint_milestones`：按 token 数保存 checkpoint（如 1B、5B、10B、14.4B tokens） 位置：`training_mode.checkpoint_milestones`，trainer 中 `total_tokens_seen >= milestone` 时触发                              | ❌ 仅 `checkpoint_every_n_steps`、`save_model_every_n_steps`（按 step） | ❌ **不支持** |
| **decay_from_pretrain 阶段** | ✅ `training_mode=decay_from_pretrain`：从 pretrain checkpoint 进入 decay 阶段，`decay_ratio` 控制 decay 阶段比例（如 10% 的 token 做 LR decay） 位置：`decay_from_pretrain.yaml`、`TokenModelTrainer` / `MSE_trainer` | ❌ 仅 pretrain + finetune，无单独 decay 阶段                              | ❌ **不支持** |
| **按 token 计数终止训练**         | ✅ 可用 `max_tokens` 或 `max_steps` 配合 `tokens_per_sample` 控制总 token 数                                                                                                                              | ❌ 仅 `max_steps`（按 step）                                           | ❌ **不支持** |
| **decay_ratio**            | ✅ `decay_ratio`（如 0.1）表示 decay 阶段占总 token 的比例                                                                                                                                                   | ❌ 无此概念                                                            | ❌ **不支持** |
| **学习率调度**                  | ✅ cosine、linear、token-based 等                                                                                                                                                                   | ✅ noop、myle、cosine、wsd、polynomial                                 | ✅ 支持      |
| **断点恢复**                   | ✅ 支持，含 `_resume_start_index`、`_saved_milestones` 等                                                                                                                                              | ✅ `FileCheckpointManager`、`trainer.restore()`                     | ✅ 支持      |
| **预训练 / 微调**               | ✅ pretrain、decay_from_pretrain、finetune                                                                                                                                                         | ✅ pretrain、finetune                                               | ✅ 支持      |
| **W&B 记录**                 | ✅ 支持                                                                                                                                                                                            | ✅ `LCMWandBRecorder`，需安装 wandb                                    | ✅ 支持      |
| **HuggingFace Hub 上传**     | ✅ `hf_repo_id`、`hf_checkpoint_subfolder` 等                                                                                                                                                      | ❌ 无                                                               | ❌ **不支持** |


---

## 三、LCM 目前做不到的能力汇总

### 预处理

1. ~~**智能引号等 Unicode 标点 → ASCII**~~：✅ 已实现（`prepare_fine_web.py` + `preprocess_filter_utils.py`）。
2. ~~**UNK 字符过滤/替换**~~：✅ 已实现（`drop_text_with_unk_tokens`，`enable_unk_filter=True` 默认开启）。
3. **Tensor 缓存**：LCM 无此机制，直接写 Parquet。

### 训练

1. **Token 里程碑 checkpoint**：LCM 按 step 保存，不支持按 token 数（如 1B、5B、10B）保存。
2. **decay_from_pretrain 阶段**：LCM 无 pretrain → decay 的专门阶段，无法做 scaling law 中常见的 decay 实验。
3. **decay_ratio**：LCM 无 decay 阶段占总 token 比例的概念。
4. **按 token 数终止训练**：LCM 仅支持 `max_steps`。
5. **HuggingFace Hub 上传**：LCM 无。

---

## 四、简要结论


| 类别      | LCM 缺失的能力                                                                 |
| ------- | ------------------------------------------------------------------------- |
| **预处理** | ~~智能引号/Unicode 标点归一化~~、~~UNK 过滤~~、Tensor 缓存                               |
| **训练**  | Token 里程碑 checkpoint、decay_from_pretrain、decay_ratio、按 token 终止、HF Hub 上传 |


若要在 LCM 上做 **scaling law 实验**（如按 token 里程碑 checkpoint、pretrain → decay 两阶段），需要：

- 在 LCM 中增加 `checkpoint_milestones`（按 token 计数）逻辑；
- 增加 `decay_from_pretrain` 训练模式（从 pretrain checkpoint 加载，仅用 decay 阶段 LR）；
- ~~在预处理中增加 `normalize_unicode_punctuation` 等文本清洗（可选）~~ ✅ 已完成。

---

## 五、任务列表（完成进度）

### 预处理端


| 任务                        | 状态        | 说明                                                                                   |
| ------------------------- | --------- | ------------------------------------------------------------------------------------ |
| 智能引号等 Unicode 标点 → ASCII  | ✅ **已完成** | `scripts/utils/preprocess_filter_utils.py` + `prepare_fine_web.py` 第 477–479 行       |
| UNK 字符过滤                  | ✅ **已完成** | `drop_text_with_unk_tokens`，`prepare_fine_web.py` 第 492–499 行，`enable_unk_filter` 参数 |
| Tensor 缓存（sentence-level） | ❌ 未完成     | LCM 仍直接写 Parquet，无 embedding 缓存机制                                                    |


### 训练端


| 任务                     | 状态    | 说明                                                                               |
| ---------------------- | ----- | -------------------------------------------------------------------------------- |
| Token 里程碑 checkpoint   | ❌ 未完成 | 需在 `lcm/train/trainer.py` 中增加 `checkpoint_milestones`，按 `total_tokens_seen` 触发保存 |
| decay_from_pretrain 阶段 | ❌ 未完成 | 需新增 recipe（如 `recipes/train/decay_from_pretrain/`）及 trainer 逻辑                   |
| decay_ratio            | ❌ 未完成 | 依赖 decay_from_pretrain，表示 decay 阶段占总 token 比例                                    |
| 按 token 数终止训练          | ❌ 未完成 | 需支持 `max_tokens` 或等效配置，替代/补充 `max_steps`                                         |
| HuggingFace Hub 上传     | ❌ 未完成 | 需在 checkpoint 保存后增加 HF 上传逻辑                                                      |


### 汇总


| 类别      | 已完成                   | 未完成                                                       |
| ------- | --------------------- | --------------------------------------------------------- |
| **预处理** | 2（Unicode 归一化、UNK 过滤） | 1（Tensor 缓存）                                              |
| **训练**  | 0                     | 5（token milestones、decay 阶段、decay_ratio、按 token 终止、HF 上传） |


