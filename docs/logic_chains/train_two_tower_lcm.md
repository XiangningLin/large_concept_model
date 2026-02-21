# Two-Tower Diffusion LCM 训练逻辑链

本文档描述 **Two-Tower Diffusion LCM** 从入口到单步训练与 checkpoint 的代码逻辑链；与 Base LCM 的差异集中在 **Criterion、模型 forward、噪声调度与 step 采样**。

---

## 1. 入口与配置

```
python -m lcm.train launcher=standalone +pretrain=two_tower ++trainer.output_dir=...
                   或
python -m lcm.train +finetune=two_tower ++trainer.model_config_or_name=my_pretrained_two_tower ...
```

- **入口**：与 Base LCM 相同，`lcm/train/__main__.py` → Hydra 合并 `recipes/train/pretrain/two_tower.yaml` 或 `finetune/two_tower.yaml`。
- **得到 Trainer**：`_trainer_` 指向 **prepare_two_tower_diffusion_lcm_trainer** → **DiffusionLCMTrainerBuilder(config).build_trainer()**。
- **DiffusionLCMTrainerBuilder** 继承 **LCMTrainerBuilder**，仅覆盖 **model_loader** → `load_two_tower_diffusion_lcm_model`，以及 **criterion** 使用 **TowerDiffusionLCMCriterionConfig**（two_tower_diffusion_next_sent / two_tower_diffusion_next_sent_finetuning）。

---

## 2. Builder 构建流程（与 Base 的差异）

- **load_data()**：与 Base 相同，仍为 **LCMDataLoader**，输出 **LCMInput**。
- **create_model()**：`create_model_config` 使用 arch（如 `two_tower_diffusion_lcm_1_6B`）或 `model_config_or_name`；**model_loader** 为 **load_two_tower_diffusion_lcm_model**，得到 **TwoTowerDiffusionLCModel**（encoder_frontend、context_encoder、denoiser、noise_scheduler、sonar_normalizer）。
- **maybe_load_model / maybe_freeze_parameters / wrap_model**：与 Base 相同；若 finetune 且无 checkpoint，从 `model_config_or_name` 加载预训练权重。
- **Criterion**：`setup_criterion()` 通过 CriterionsFactory 构建 **TwoTowerDiffusionCriterion**（或 finetuning 变体），依赖 model 上的 **noise_scheduler** 与 config 中的 **step_sampling**、**cf_guidance_probability**。

---

## 3. 单步训练逻辑（与 Base 共用 Trainer.run / _train_step）

- 与 Base LCM 使用同一 **Trainer** 基类；`_train_step` 流程一致：取 batch → `loss = self.criterion(batch)` → backward → 梯度处理 → optimizer step → lr_scheduler.step() → 按条件 checkpoint / validate / publish_metrics。
- 唯一区别是 **self.criterion** 为 **TwoTowerDiffusionCriterion**，且 **model** 为 **TwoTowerDiffusionLCModel**，forward 需要 **(input_batch, noisy_target_batch, cf_guidance_prob)**。

---

## 4. Criterion 链（Two-Tower Diffusion）

**TwoTowerDiffusionCriterion**（`lcm/train/two_tower_diffusion_lcm/criterion.py`）在构造时：

- 从 **base_model** 取 **noise_scheduler**（DDIM）；取 **prediction_type**（sample / epsilon / v_prediction）。
- 校验 **cf_guidance_probability** 与 model 的 **trained_with_cf_guidance** 一致。
- 构建 **StepsSampler**（step_sampling config + noise_scheduler），用于采样 diffusion timestep 与可选 loss 权重。

**__call__(batch: LCMInput)** 流程：

```
__call__(batch)
    │
    ├─ prepare_input_and_mask(batch)
    │     → input_embeddings = batch.prepare_input(style)
    │     → 可选：input_embeddings = input_embeddings.normalize_seqs(sonar_normalizer)
    │     → target_mask = ones 且与 padding 对齐
    │     → 返回 input_embeddings, target_batch(=input_embeddings.clone()), target_mask
    │
    ├─ sample_noisy_input_and_targets(target_batch, target_mask)
    │     → timesteps = step_sampler.sample(size=(B,T), device=...)
    │     → noise_seqs = randn_like(input_seqs)
    │     → 按 prediction_type 计算 target（clean / epsilon / velocity）
    │     → noisy_input_seqs = noise_scheduler.add_noise(clean, noise, timesteps)
    │     → noisy_target_batch = EmbeddingsBatch(noisy_input_seqs, padding_mask, diffusion_timesteps=timesteps)
    │     → 返回 noisy_target_batch, target, target_mask
    │
    ├─ output_batch = self.model(input_batch, noisy_target_batch, cf_guidance_prob=self.cf_guidance_probability)
    │     → TwoTowerDiffusionLCModel.forward(batch, noisy_batch, cf_guidance_prob)
    │       （context encoder 编码 input_batch；denoiser 在 noisy_batch 上做去噪预测）
    │
    ├─ flattened_predictions = output_batch.seqs.view(-1, C)[target_mask]
    │   flattened_target     = target[target_mask]
    │
    ├─ compute_loss(flattened_predictions, flattened_target)
    │     → reconstruction_loss, plain_reconstruction_loss, unnormalized_reconstruction_loss
    │     → 可选 step bucket 日志（log_losses_per_timestep_bucket）
    │
    ├─ 可选：gammas = step_sampler.get_loss_scales(batch_steps)；reconstruction_loss *= gammas
    │
    └─ 按 reduction 聚合 → LossTerm(value, num_target_elements, summands)
```

- **Step 采样**：`StepsSampler`（`lcm/train/step_sampler.py`）根据 config（如 `sampling: "uniform"`, `weighting: "none"`）在 `[0, num_diffusion_train_steps)` 上采样 timestep；可选对 loss 按 step 加权（如 SNR weighting）。
- **Classifier-free guidance**：forward 时以概率 `cf_guidance_probability` 将 conditioning 置空，模型需在构建时 `trained_with_cf_guidance=True`。

---

## 5. 模型 Forward（TwoTowerDiffusionLCModel）

- **签名**：`forward(batch: EmbeddingsBatch, noisy_batch: EmbeddingsBatch, cf_guidance_prob: float = 0.0) -> EmbeddingsBatch`。
- **含义**：
  - **batch**：干净 context（source 或 source+target 的 embedding 序列），用于 **context_encoder** 编码。
  - **noisy_batch**：加噪后的 target 序列（带 **diffusion_timesteps**），送入 **denoiser** 预测 clean / noise / velocity。
- **输出**：与 noisy_batch 同 shape 的 **EmbeddingsBatch**，表示对每个位置的去噪预测；criterion 再与 **target**（由 prediction_type 决定）做 MSE。

---

## 6. 数据流小结（Two-Tower）

| 阶段 | 数据类型 | 说明 |
|------|----------|------|
| DataLoader | 同 Base | LCMInput |
| prepare_input_and_mask | LCMInput → input_batch, target_batch, target_mask | 可选 normalizer |
| sample_noisy_input_and_targets | target_batch, target_mask → noisy_target_batch, target, target_mask | timestep 采样、加噪、按 prediction_type 算 target |
| Model forward | input_batch, noisy_target_batch, cf_guidance_prob → EmbeddingsBatch | context encoder + denoiser |
| Criterion | 取 output_seqs 与 target 在 target_mask 上 MSE，可选 step 加权 → LossTerm | two_tower_diffusion_next_sent |

---

## 7. Checkpoint 与恢复

- 与 Base LCM 相同：**FileCheckpointManager**、**trainer.restore()** 当存在 checkpoint 时自动恢复；state 含 model、optimizer、lr_scheduler、step_nr 等；Two-Tower 的 **noise_scheduler**、**denoiser** 等均在 model 内，一并保存/加载。

---

## 8. 涉及文件一览

| 文件 | 作用 |
|------|------|
| `lcm/train/two_tower_diffusion_lcm/trainer.py` | TwoTowerDiffusionLCMTrainingConfig、DiffusionLCMTrainerBuilder、prepare_two_tower_diffusion_lcm_trainer |
| `lcm/train/two_tower_diffusion_lcm/criterion.py` | TowerDiffusionLCMCriterionConfig、TwoTowerDiffusionCriterion、sample_noisy_input_and_targets、compute_loss |
| `lcm/train/step_sampler.py` | StepsSampler、StepsSamplerConfig（采样 timestep、可选 loss scale） |
| `lcm/models/two_tower_diffusion_lcm/loader.py` | load_two_tower_diffusion_lcm_model |
| `lcm/models/two_tower_diffusion_lcm/builder.py` | TwoTowerDiffusionLCModel、forward(batch, noisy_batch, cf_guidance_prob) |
| `lcm/nn/schedulers/ddim.py` | DDIMScheduler（add_noise、get_velocity、num_diffusion_train_steps、prediction_type） |
| `lcm/nn/denoisers/` | LCMDenoiser（在 noisy 序列上做去噪预测） |
| `recipes/train/pretrain/two_tower.yaml`, `two_tower_780M.yaml` | 预训练 recipe |
| `recipes/train/finetune/two_tower.yaml` | 微调 recipe（criterion: two_tower_diffusion_next_sent_finetuning） |

（与 Base 共用的 Trainer、DataLoader、batch、optim、metrics 等见 `train_base_lcm.md`。）
