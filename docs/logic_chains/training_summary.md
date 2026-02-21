# LCM 训练能力总结（Base LCM 与 Two-Tower Diffusion LCM）

本文档概括 **LCM-base** 与 **LCM-two-tower** 在官方实现中支持的训练能力，便于快速判断「能实现怎样的训练」。

---

## 1. 学习率调度（Learning Rate Schedule）

| 调度类型 | 配置名 | 说明 |
|----------|--------|------|
| **noop** | `lr_schedule: noop` | 无调度，始终使用初始学习率 |
| **myle** | `lr_schedule: myle` | Fairseq 风格的 inv-sqrt 调度 + warmup |
| **cosine** | `lr_schedule: cosine` | Cosine 退火（warmup + cosine decay to `final_lr`） |
| **wsd** | `lr_schedule: wsd` | Warmup–Stable–Decay 三阶段（TriStageLR），由 `lr_stage_ratios` 控制比例 |
| **polynomial** | `lr_schedule: polynomial` | 多项式衰减 |

- 配置位置：`TrainingConfig`（`lcm/train/trainer.py`）及各 recipe YAML。
- 实现：`lcm/train/optim.py` 的 `build_lr_scheduler()`，使用 fairseq2 的 `MyleLR`、`CosineAnnealingLR`、`TriStageLR`、`PolynomialDecayLR` 等。
- 常用参数：`lr`、`start_lr`、`final_lr`、`num_lr_warmup_steps`、`lr_stage_ratios`（仅 wsd）、`max_steps`。

**结论**：**支持 cosine 与 WSD**，以及 noop、myle、polynomial；recipe 中预训练/微调多用 **cosine**。

---

## 2. 断点恢复（Checkpoint Resume）

- **支持**：支持从已有 checkpoint 恢复训练。
- **机制**：
  - 使用 `FileCheckpointManager`（fairseq2），checkpoint 目录为 `output_dir/checkpoints`。
  - 在 **Builder.build_trainer()** 中：`self.has_checkpoint = checkpoint_manager.has_checkpoint()`；若为 True，在创建并 setup 完 Trainer 后调用 **trainer.restore()**。
  - `restore()` 会 `load_last_checkpoint()`，得到 `step_nr` 和 state，然后 `load_state_dict(checkpoint)`，并将 `self.step_nr = step_nr + 1`，从而从下一 step 继续。
- **保存内容**：optimizer、lr_scheduler、model、rng_bag 等通过 `Trainer.state_dict()` 保存；metadata 含 config 与可选的 crash 信息。
- **结论**：**支持断点恢复**，只要 `output_dir/checkpoints` 下存在上次保存的 checkpoint，再次用相同 `output_dir` 启动即可自动恢复。

---

## 3. 预训练（Pretrain）与微调（Finetune）

| 能力 | Base LCM | Two-Tower LCM |
|------|----------|----------------|
| **预训练** | 支持 | 支持 |
| **微调** | 支持 | 支持 |

- **预训练**：
  - 使用 **model_arch**（如 `base_lcm_1_6B`、`two_tower_diffusion_lcm_1_6B`）从随机初始化开始训练。
  - Recipe：`+pretrain=mse` / `mse_780M`（Base）、`+pretrain=two_tower` / `two_tower_780M`（Two-Tower）。
- **微调**：
  - 使用 **model_config_or_name** 指向已注册的 fairseq2 模型名（或 model card），从预训练权重加载后再训练。
  - 在 Builder 中：`create_model_config(set_finetune_flag=True)` 会设 `self.finetune = True`；`maybe_load_model(model)` 在 **无 checkpoint 且 finetune=True** 时从 `model_config_or_name` 加载权重并 broadcast 到各 rank。
  - Recipe：`+finetune=mse`、`+finetune=two_tower`，并覆盖 `model_config_or_name` 为你的预训练模型名。

**结论**：**两者均支持预训练与微调**；微调时需先在 fairseq2 中注册预训练 checkpoint（如 model card），再在 recipe 中指定 `model_config_or_name`。

---

## 4. W&B（Weights & Biases）记录

- **支持**：支持使用 WandB 记录训练指标。
- **实现**：
  - `lcm/train/metrics.py` 中的 **LCMWandBRecorder**；在 **rank 0** 与 `LogMetricRecorder`、`TensorBoardRecorder` 一起加入 `Trainer.metric_recorders`。
  - 若未安装 `wandb`，会打 warning 并跳过 W&B，不影响训练。
- **配置**（`TrainingConfig`）：
  - `wandb_project`：项目名，默认 `"uncategorized"`。
  - `wandb_run_name`：run 名称。
  - `wandb_entity`：可选 entity。
- **记录内容**：与 TensorBoard 一致，由 `record_metrics()` 写入（如 loss、lr、grad_norm、elements_per_second 等）；`resume="allow"` 支持断点后继续写同一 run。

**结论**：**有 WandB 记录**；需安装 `wandb` 并（可选）`wandb login`，通过 `wandb_project` / `wandb_run_name` 等配置即可。

---

## 5. 其他训练相关能力（共用）

- **分布式**：支持 **FSDP**（默认）与 **DDP**；单卡时可不包装。
- **混合精度**：支持 `dtype=torch.float16` + `DynamicLossScaler`；可选 `use_autocast`。
- **梯度**：梯度裁剪（`max_grad_norm`）、可选梯度归一化（按 target 数量或 world size）。
- **验证**：按 `validate_every_n_steps` 做验证，指标写入 TensorBoard / WandB。
- **Checkpoint 策略**：`checkpoint_every_n_steps`、`save_model_every_n_steps`（FSDP 下 consolidate 并保存 `model.pt`）、`keep_last_n_checkpoints`、`preserve_consolidated_models`。
- **冻结**：`freezing_strategy`（none / modules / ffn / adaln / ffn-adaln）、`freeze_modules` 列表。
- **OOM/NaN 处理**：可选重试（`raise_oom` / `raise_nan_or_inf` 为 False 时），重试前会 checkpoint；超过 `max_ooms` / `max_nans_or_infs` 再抛出。

---

## 6. Base LCM 与 Two-Tower 在训练上的主要区别

| 项目 | Base LCM | Two-Tower LCM |
|------|----------|----------------|
| **Criterion** | `next_sentence_mse`（预训练）、`target_mse`（微调） | `two_tower_diffusion_next_sent`（预训练）、`two_tower_diffusion_next_sent_finetuning`（微调） |
| **Loss** | MSE 重建（next sentence 或 source→target） | 扩散去噪目标（MSE on 预测的 clean/noise/velocity，可选 step 加权） |
| **模型 forward** | 单 batch：`model(batch)` → 预测下一句 embedding | 双输入：`model(input_batch, noisy_target_batch, cf_guidance_prob)`，含加噪与去噪 |
| **额外组件** | 无 | noise_scheduler（DDIM）、step_sampler（采样 diffusion step）、可选 classifier-free guidance |

上述能力在逻辑链文档中有更细的代码级说明：`train_base_lcm.md`、`train_two_tower_lcm.md`。
