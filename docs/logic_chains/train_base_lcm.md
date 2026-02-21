# Base LCM 训练逻辑链

本文档描述 **Base LCM**（MSE 重建）从入口到单步训练与 checkpoint 的代码逻辑链。

---

## 1. 入口与配置

```
python -m lcm.train launcher=standalone +pretrain=mse ++trainer.output_dir=...
                   或
python -m lcm.train launcher=submitit +pretrain=mse ...
```

- **入口**：`lcm/train/__main__.py` → `main(config)` → `asyncio.run(run(config))`。
- **Hydra**：`config_path="../../recipes/train"`，`config_name="defaults.yaml"`；`+pretrain=mse` 会合并 `recipes/train/pretrain/mse.yaml`（或 `mse_780M` 等）。
- **得到 Trainer**：`get_trainer(train_config)`（`lcm/train/common.py`）→ `_parse_training_config(train_config)` 解析 `_trainer_` 指向的 callable → 对 Base LCM 为 **prepare_lcm_trainer** → 返回 `LCMTrainerBuilder(config).build_trainer()`。

---

## 2. Builder 构建流程（LCMTrainerBuilder.build_trainer）

```
LCMTrainerBuilder.build_trainer()
    │
    ├─ load_data()
    │     → LCMDataLoader(training_data, validation_data, data_loading_config, max_subword_length, dtype, gang)
    │     → 从 datacards 按 name 解析 Parquet 路径，构建 weighted pipeline，返回 (training_data_loader, validation_data_loader)
    │
    ├─ FileCheckpointManager(output_dir/checkpoints, gang)
    ├─ has_checkpoint = checkpoint_manager.has_checkpoint()
    │
    ├─ create_model()
    │     → create_model_config(set_finetune_flag=True)  # 若 model_config_or_name 为已注册名 → finetune=True
    │     → model_loader._factory(model_config, device, dtype)  # load_base_lcm_model
    │     → 得到 BaseLCModel（frontend + lcm decoder + postnet）
    │
    ├─ maybe_load_model(model)
    │     → 若 has_checkpoint=False 且 finetune=True：从 model_config_or_name 加载预训练 state_dict，broadcast 到各 rank
    │
    ├─ maybe_freeze_parameters(model)
    │     → 按 freezing_strategy 冻结 FFN / adaln / 指定 modules
    │
    ├─ wrap_model_with_fsdp(model) 或 wrap_model_with_ddp(model)
    │
    └─ LCMTrainer(..., checkpoint_manager, ...).setup()
          → 若 has_checkpoint：trainer.restore()  # 加载 last checkpoint，step_nr = loaded_step + 1
```

- **数据**：训练/验证数据来自 `training_data` / `validation_data` 的 `name`（如 `pretraining_data=train`），在 datacards 中解析 parquet_path；batch 格式为 **LCMInput**（含 source/target embeddings、padding、doc lengths 等），由 `LCMDataLoader.iterate_batches()` 产出。
- **模型**：`load_base_lcm_model`（`lcm/models/base_lcm/loader.py`）根据 arch 或 config 构建 **BaseLCModel**；forward 输入为 **EmbeddingsBatch**，输出亦为 **EmbeddingsBatch**（预测的下一句 embedding）。

---

## 3. 单步训练逻辑（Trainer._train_step）

```
_train_step(data_iter)
    │
    ├─ 取 gradient_accumulation 个 batch：batches = [next(data_iter), ...]
    │     batch 类型：LCMInput（见 lcm/datasets/batch）
    │
    ├─ train_metric_bag.begin_updates()
    │
    ├─ for batch in batches:
    │     ├─ loss = self.criterion(batch)   # 见下节
    │     ├─ train_metric_bag.update([loss])
    │     ├─ loss_scaler.backward(loss.value)
    │     └─ (no_sync 用于梯度累积时除最后一步外不同步)
    │
    ├─ process_gradients(step_nr, num_targets)
    │     → 梯度裁剪、可选归一化、unscale、clip_gradient_norm、跨 rank 一致性检查
    │
    ├─ loss_scaler.run_optimizer_step(step_nr)
    │     → 若 overflow：rollback 指标并可能重做本 step；否则：
    │     → lr_scheduler.step()
    │     → stepped = True
    │
    ├─ optimizer.zero_grad(set_to_none=True)
    │
    └─ train_metric_bag.commit_updates()；记录 grad_norm、wall_time 等；按条件 publish_metrics / checkpoint / validate
```

---

## 4. Criterion 链（Base LCM）

**预训练** 使用 `next_sentence_mse`，**微调** 使用 `target_mse`；均由 **ReconstructionCriterion** / **TargetMSECriterion**（`lcm/train/mse_lcm/criterion.py`）实现。

```
CriterionsFactory.build_criterion(name, config, model)
    → ReconstructionCriterion 或 TargetMSECriterion
```

**ReconstructionCriterion(batch: LCMInput)** 流程：

```
__call__(batch)
    │
    ├─ prepare_input_and_mask(batch)
    │     → batch.prepare_input(style=UNSUPERVISED/SUPERVISED)   # 得到 EmbeddingsBatch (B, T, C)
    │     → batch.prepare_target_mask(..., min_context_size)    # 得到 target_mask (B, T, C) 或 (B, T)
    │     → 可选：input_embeddings = input_embeddings.normalize_seqs(sonar_normalizer)
    │
    ├─ output_embeddings = self.model(input_embeddings)
    │     → BaseLCModel.forward(batch): frontend → lcm decoder → postnet → EmbeddingsBatch
    │
    ├─ 构造 target / prediction（shift 对齐 next-sentence）：
    │     target_seqs     = input_embeddings.seqs[:, 1:]   # 真值 s2, s3, ...
    │     predicted_seqs  = output_embeddings.seqs[:, :-1]  # 预测 s<=1, s<=2, ...
    │     target_mask     = target_mask[:, 1:].reshape(-1)
    │
    ├─ flattened_predictions = predicted_seqs.view(-1, C)[target_mask]
    │   flattened_target     = target_seqs.view(-1, C)[target_mask]
    │
    ├─ compute_loss(flattened_predictions, flattened_target)
    │     → compute_standard_mse(...) → reconstruction_loss, mse_loss
    │
    └─ 按 reduction（sum/mean）聚合 → LossTerm(value, num_target_elements, summands)
```

- **LCMInput**：由 dataloader 从 Parquet 读出并封装为带 source/target embedding 序列、padding mask、doc lengths 等；`prepare_input` 根据 LCMStyle 取 source 或 concat source+target 等。
- **target_mse**：与上面同一套，但 style=SUPERVISED，且使用 source/target 对的 target 部分作为预测目标（用于有监督微调）。

---

## 5. 数据流小结（Base LCM）

| 阶段 | 数据类型 | 说明 |
|------|----------|------|
| Datacard / Recipe | `training_data` / `validation_data` 的 name | 指向 datacards 中的 parquet_path 与列名 |
| DataLoader | Parquet → 句子级 embedding 序列 → batching / packing | 输出 **LCMInput** |
| prepare_input_and_mask | LCMInput → EmbeddingsBatch, target_mask | 可选 normalizer |
| Model forward | EmbeddingsBatch → EmbeddingsBatch | BaseLCModel: frontend → decoder → postnet |
| Criterion | 对齐 shift → MSE on masked positions → LossTerm | next_sentence_mse / target_mse |

---

## 6. Checkpoint 与恢复

- **保存**：`_checkpoint()` 调用 `checkpoint_manager.save_state(self.state_dict(), ...)`，state 含 model、optimizer、lr_scheduler、step_nr、rng_bag 等；FSDP 时还可 `save_consolidated_fsdp_model` 得到单文件 `model.pt`；并写 `model_card.yaml`。
- **恢复**：`restore()` 调用 `checkpoint_manager.load_last_checkpoint()` → `self.load_state_dict(checkpoint)`，`self.step_nr = step_nr + 1`；下次 `run()` 从该 step 继续循环。

---

## 7. 涉及文件一览

| 文件 | 作用 |
|------|------|
| `lcm/train/__main__.py` | Hydra 入口，调用 get_trainer 并（通过 launcher）run |
| `lcm/train/common.py` | get_trainer、解析 _trainer_ |
| `lcm/train/trainer.py` | TrainingConfig、Trainer（run、_train_step、checkpoint、restore）、TrainerBuilder |
| `lcm/train/lcm/trainer.py` | LCMTrainingConfig、LCMTrainer、LCMTrainerBuilder、prepare_lcm_trainer |
| `lcm/train/mse_lcm/criterion.py` | ReconstructionCriterion、TargetMSECriterion |
| `lcm/train/criterion.py` | Criterion 基类、CriterionsFactory |
| `lcm/train/optim.py` | build_lr_scheduler（noop/myle/cosine/wsd/polynomial） |
| `lcm/datasets/dataloading.py` | LCMDataLoader、pipeline、iterate_batches |
| `lcm/datasets/batch.py` | LCMInput、EmbeddingsBatch、prepare_input、prepare_target_mask |
| `lcm/models/base_lcm/loader.py` | load_base_lcm_model |
| `lcm/models/base_lcm/builder.py` | BaseLCModel、forward |
| `recipes/train/pretrain/mse.yaml`, `mse_780M.yaml` | 预训练 recipe |
| `recipes/train/finetune/mse.yaml` | 微调 recipe |
