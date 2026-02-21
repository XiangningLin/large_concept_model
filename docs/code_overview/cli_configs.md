# LCM 项目 CLI 参数完整参考

本文档覆盖所有 `recipes/train/` 配置参数，按**功能模块**分块说明。

---

## 快速选择：Recipe 入口

| 训练模式 | Hydra 选项 | Recipe 文件 |
|----------|-----------|-------------|
| Pretrain 1.6B | `+pretrain=mse` | [pretrain/mse.yaml](file:///projects/bfaq/jlyu3/large_concept_model/recipes/train/pretrain/mse.yaml) |
| Pretrain 780M | `+pretrain=mse_780M` | [pretrain/mse_780M.yaml](file:///projects/bfaq/jlyu3/large_concept_model/recipes/train/pretrain/mse_780M.yaml) |
| Decay-from-Pretrain | `+decay_from_pretrain=mse` | [decay_from_pretrain/mse.yaml](file:///projects/bfaq/jlyu3/large_concept_model/recipes/train/decay_from_pretrain/mse.yaml) |
| Finetune | `+finetune=mse` | `finetune/mse.yaml` |

---

## 模块 1：输出与身份

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `??`（必填） | 本地 checkpoint / log / WandB 输出目录 |
| `training_mode` | `"pretrain"` | `"pretrain"` 或 `"decay_from_pretrain"`；决定 LR 调度逻辑 |
| `seed` | `1`（recipe）/ `2`（代码默认） | 全局随机种子（Python / NumPy / PyTorch / CUDA） |
| `dtype` | `"torch.float16"` | 模型计算精度；`float32` 兼容性更好但更慢 |
| `debug` | `false` | 开启 ATEN / NCCL debug 日志 |
| `experiment_name` | `null` | 作业追踪名；null 时使用默认命名 |

---

## 模块 2：模型架构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_arch` | `base_lcm_1_6B` | fairseq2 asset 名；决定参数量和层数 |
| `model_arch_overrides` | `null` | 覆写 `model_arch` 中的部分参数（dict） |
| `model_config_or_name` | `null` | 直接指定模型配置；设置时忽略 `model_arch` |
| `criterion.name` | `next_sentence_mse` | 损失函数；`next_sentence_mse` 或 `two_tower_diffusion` |
| `criterion.reduction` | `sum` | 损失归一化方式 |
| `criterion.compute_rmse` | `false` | 是否额外计算 RMSE 指标 |

---

## 模块 3：优化器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | `0.0004`（recipe） | 峰值学习率（WSD 稳定期 LR） |
| `weight_decay` | `0.1` | AdamW 正则化 |
| `adam_betas` | `[0.9, 0.95]`（recipe）/ `[0.9, 0.98]`（代码） | Adam 一阶/二阶动量 |
| `adam_eps` | `1e-5`（recipe）/ `1e-6`（代码） | 数值稳定项 |
| `max_grad_norm` | `25`（recipe）/ `1000`（代码） | 梯度裁剪阈值 |
| `turn_off_grad_normalization` | `false` | 为 `true` 时完全关闭梯度裁剪 |
| `gradient_accumulation` | `1` | 梯度累积步数；等效 batch size × N |
| `use_optimizer_in_fp32` | `true` | 优化器状态保持 fp32 |
| `use_autocast` | `false` | 开启 AMP autocast；通常不需要（已用 fp16） |
| `loss_scaler_init_scale` | `32768.0`（2¹⁵） | 动态 loss scaler 初始 scale |
| `loss_scaler_scale_window` | `null` | scaler 更新之前连续无溢出步数；null = 用默认 |

---

## 模块 4：学习率调度（WSD）

> 当前所有 recipe 使用 **WSD（Warmup-Stable-Decay）** 调度器。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr_schedule` | `wsd` | 调度器类型；支持 `noop`、`myle`、`cosine`、`wsd` |
| `num_lr_warmup_steps` | `10000`（pretrain recipe） | 固定 warmup 步数；Pretrain 模式下与 `lr_stage_ratios[0]` **独立叠加**；Decay 模式下通常设 0 |
| `lr_stage_ratios` | `[0.1, 0.8, 0.1]`（recipe）<br>`[0.1, 0.4, 0.5]`（代码默认） | WSD 三阶段比例：Warmup / Stable / Decay，必须加和为 1.0 |
| `start_lr` | `1e-7` | Warmup 起始 LR |
| `final_lr` | `1e-5` | Decay 结束 LR |
| `max_steps` | `250000` | 总训练步数；Decay 模式自动计算，无需手动设置 |

> [!NOTE]
> **`num_lr_warmup_steps` vs `lr_stage_ratios[0]`**：两者都控制 warmup，但`num_lr_warmup_steps` 是固定步数，`lr_stage_ratios[0]` 是占总步数的比例。WSD 代码中 warmup 以 `num_lr_warmup_steps` 为准（忽略 `lr_stage_ratios[0]`）。

**Pretrain 典型 `lr_stage_ratios`**：
```yaml
lr_stage_ratios: [0.1, 0.8, 0.1]  # 10% warmup → 80% stable → 10% decay
```

**Decay 模式（自动强制）**：
```yaml
lr_stage_ratios: [0.0, 0.0, 1.0]  # 100% decay，从 peak LR 立即衰减
```

---

## 模块 5：Decay-from-Pretrain 专用

> 仅在 `training_mode: "decay_from_pretrain"` 时有效。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pretrain_total_steps` | `250000` | Pretrain 实际跑的步数（从 checkpoint 信息获取） |
| `decay_ratio` | `0.1` | Decay 占总预算 S 的比例；`decay_steps = pretrain_steps × ratio / (1 - ratio)` |
| `num_lr_warmup_steps` | `0` | Decay 前短暂 warmup；通常设 0 |
| `pretrain_checkpoint_path` | `null` | 本地 pretrain checkpoint 路径（与 HF Hub 二选一） |

**公式**：`pretrain_total_steps=100000, decay_ratio=0.1` → `decay_steps ≈ 11111`，`max_steps` 自动设为 `111111`

**状态继承策略**：

| 状态 | 继承？ |
|------|--------|
| `model` | ✅ |
| `optimizer` | ✅ |
| `training_data_loader`（数据进度） | ✅ |
| `rng_bag`（随机状态） | ✅ |
| `step_nr` | ✅（从 pretrain 继续计数） |
| `lr_scheduler` | ❌（使用 decay-only WSD） |
| `saved_milestones` | ❌（使用 decay recipe 的新 milestones） |

---

## 模块 6：Token 里程碑 Checkpoint

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `checkpoint_milestones` | `null` | Token 数列表；达到时额外保存 checkpoint |
| `tokens_per_sentence` | `18.5` | 每个 SONAR 句向量对应的平均子词 token 数 |

```yaml
checkpoint_milestones:
  - 1_000_000_000   # 1B tokens
  - 5_000_000_000   # 5B tokens
  - 16_000_000_000  # 16B tokens
```

---

## 模块 7：常规 Checkpoint 策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `checkpoint_every_n_steps` | `2000` | 每 N 步保存一次完整 checkpoint（含 optimizer 状态、数据进度、RNG 状态） |
| `save_model_every_n_steps` | `2000` | 每 N 步保存一次 consolidated 模型权重（用于推理） |
| `keep_last_n_checkpoints` | `2` | 保留最新 N 个 checkpoint；-1 = 保留全部 |
| `preserve_consolidated_models` | `true` | 是否永久保留 consolidated 模型（不随旧 checkpoint 被删除） |
| `gc_every_n_steps` | `1000` | 每 N 步运行一次 Python GC 回收 |

---

## 模块 8：HuggingFace Hub 集成

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hf_repo_id` | `null` | HF 仓库 ID（如 `user/repo`）；设置后启用上传/下载 |
| `hf_token` | `null` | HF 访问令牌；也可通过 `$HF_TOKEN` 环境变量设置 |
| `hf_checkpoint_subfolder` | `null` | HF 仓库中**读取** checkpoint 的子文件夹 |
| `hf_checkpoint_filename` | `null` | 要下载的具体 checkpoint 文件名；`null` = 不从 HF 下载 |
| `hf_checkpoint_save_subfolder` | `null` | HF 仓库中**写入** checkpoint 的子文件夹；`null` 时复用 `hf_checkpoint_subfolder` |

**典型用法**：
- **Pretrain**：`hf_checkpoint_subfolder` 和 `hf_checkpoint_save_subfolder` 相同（存入同一目录）
- **Decay**：`hf_checkpoint_subfolder` 指向 pretrain 目录（读），`hf_checkpoint_save_subfolder` 指向 decay 目录（写）

---

## 模块 9：WandB 监控

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `wandb_project` | `null` | WandB 项目名 |
| `wandb_run_name` | `null` | Run 名；**`null` 时自动生成 `lcm_{training_mode}_{timestamp}`** |
| `wandb_entity` | `null` | WandB 账户/组织名 |
| `wandb_run_id` | `null` | 已有 Run 的 ID；设置后续接同一曲线（断点恢复 / pretrain→decay 连续曲线） |

---

## 模块 10：验证与指标

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `validate_every_n_steps` | `10000` | 每 N 步运行一次验证集评估 |
| `publish_metrics_every_n_steps` | `100` | 每 N 步上报 loss 等指标到 WandB / TensorBoard |

---

## 模块 11：数据加载（`data_loading_config`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_tokens` | `7168` | 每批最大句子数；越大显存占用越高 |
| `min_batch_size` | `1` | 最小 batch size；低于此值的 batch 被丢弃 |
| `len_to_wrap_long_seq` | `128` | 超过此句子数的文档自动截断换行 |
| `packing` | `false` | 是否将多个短文档 pack 到单个 batch 单元 |
| `min_length_of_sequences` | `1` | 序列最短句子数；过滤过短样本 |
| `min_length_after_batching` | `2` | batch 后最终最短句子数 |
| `num_parallel_calls` | `1` | Data pipeline 并行读取线程数 |
| `nb_prefetch` | `5` | 预取 batch 数；影响 GPU 利用率 |
| `nb_epochs` | `1` | 数据集循环次数 |
| `ignore_checkpointed_pipeline` | `false` | 设为 `true` 时忽略 checkpoint 中的数据进度，从头开始读数据 |

**验证集专用**（`validation_data_loading_config`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `len_to_wrap_long_seq` | `128` | 验证集截断长度 |

---

## 模块 12：数据来源（`training_data` / `validation_data`）

> Hydra list 索引语法：`++trainer.training_data.0.name="..."` 中 `.0` = 第一个数据集，`.1` = 第二个。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training_data[i].name` | `"pretraining_data=train"` | fairseq2 datacard 名，格式 `dataset=split` |
| `training_data[i].source_suffix_text` | `"End of text."` | 追加在每个样本源端的标记 |
| `validation_data[i].name` | `"pretraining_data=validation"` | 验证集 datacard 名 |

---

## 模块 13：分布式训练（FSDP / DDP）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_fsdp` | `true` | 开启 FSDP 分布式；单卡可设 `false`（使用 DDP） |
| `fsdp_fp32_reduce` | `true`（recipe） | FSDP 梯度规约时转 fp32 |
| `fsdp_wrap_granularity` | `"model"` | FSDP wrap 粒度；`"model"` / `"block"` / `"layer"` |
| `fsdp_memory_policy` | `"standard"` | `"standard"` / `"save_all_mem"` |
| `freeze_modules` | `null` | 指定冻结的模块名列表（配合 `freezing_strategy` 使用） |
| `freezing_strategy` | `"none"` | `"none"` / `"modules"` / `"ffn"` / `"adaln"` / `"ffn-adaln"` |

---

## 模块 14：计算资源（SLURM）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `requirements.nodes` | `4` | 节点数 |
| `requirements.tasks_per_node` | `8` | 每节点进程数（通常 = GPU 数） |
| `requirements.gpus_per_node` | `8` | 每节点 GPU 数 |
| `requirements.cpus_per_task` | `32` | 每进程 CPU 数 |
| `requirements.mem_gb` | `0` | 内存申请量（GB）；0 = 不限制 |
| `requirements.timeout_min` | `10000` | SLURM 超时（分钟） |

---

## 模块 15：性能分析（Profiler）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `profile` | `false` | 开启 PyTorch Profiler |
| `profiler_skip_first` | `200` | 跳过前 N 步（warmup） |
| `profiler_active` | `3` | 实际记录步数（建议 ≤10，否则 TensorBoard 无法加载） |

---

## 错误容忍

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `raise_oom` | `false` | `true` = OOM 时直接抛出；`false` = 重试 |
| `raise_nan_or_inf` | `false` | `true` = NaN/Inf 时直接抛出 |
| `max_ooms` | `10` | `raise_oom=false` 时最多容忍 OOM 次数 |
| `max_nans_or_infs` | `10` | `raise_nan_or_inf=false` 时最多容忍 NaN/Inf 次数 |

---

## 参数修改速查

| 想达到的效果 | 修改的参数 |
|-------------|-----------|
| 调大有效 batch size | `gradient_accumulation`，`data_loading_config.max_tokens` |
| 调整 LR 曲线形状 | `lr_stage_ratios`，`start_lr`，`final_lr`，`lr` |
| 调整 warmup 长度 | `num_lr_warmup_steps` |
| 更频繁保存 checkpoint | `checkpoint_every_n_steps`，`save_model_every_n_steps` |
| 在特定 token 数保存 | `checkpoint_milestones` |
| 忽略 checkpoint 数据进度（重新读数据） | `data_loading_config.ignore_checkpointed_pipeline: true` |
| 从 HF 加载 pretrain checkpoint | `hf_repo_id`，`hf_token`，`hf_checkpoint_subfolder`，`hf_checkpoint_filename` |
| 将 decay checkpoint 存到 HF 不同目录 | `hf_checkpoint_save_subfolder` |
| 断点恢复后 WandB 续接 | `wandb_run_id`（填入 pretrain 的 run ID） |
| 切换到 decay 阶段 | `training_mode: decay_from_pretrain`，`pretrain_total_steps`，`decay_ratio` |
| 冻结部分模块 | `freezing_strategy`，`freeze_modules` |
| 减少 data pipeline 卡顿 | `nb_prefetch`，`num_parallel_calls` |

---

## 附录：预处理脚本参数（`scripts/prepare_fine_web.py`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | 必填 | 输出目录 |
| `num_samples` | int\|None | `None` | 处理样本数；`None` = 全量 |
| `start_index` | int | `0` | 起始样本索引（多 GPU 并行分段用） |
| `batch_size` | int | `10` | SONAR 编码批大小 |
| `max_sentence_length` | int | `256` | 句子最大长度 |
| `add_split_column` | bool | `True` | 添加 train/validation split 列 |
| `train_ratio` | float | `0.8` | 训练集比例 |
| `seed` | int | `42` | 随机种子 |
| `checkpoint_interval` | int | `1000` | 每 N 样本保存预处理 checkpoint |
| `enable_unk_filter` | bool | `True` | 过滤含 SONAR UNK 的句子 |
