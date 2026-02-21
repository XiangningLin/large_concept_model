# LCM 功能模块测试指南

所有测试使用 **mse_60M** 模型（~62M参数），基于 `output/fine_web_test`（100条样本）的 toy 数据。每个测试都是一行 CLI 命令，可以直接复制粘贴运行。

> **前置条件**: 确保 `datacards.yaml` 中 `pretraining_data` 指向 `output/fine_web_test`，且虚拟环境可用。

---

## 测试 1: Checkpoint 保存与恢复

**目的**: 验证 checkpoint 能正确保存，训练状态（model weights、optimizer、step_nr）能完全恢复并继续训练。

**步骤 1 — 训练 10 步，每 5 步保存 checkpoint:**
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_ckpt ++trainer.experiment_name=test_ckpt ++trainer.data_loading_config.max_tokens=500 ++trainer.max_steps=10 ++trainer.checkpoint_every_n_steps=5 ++trainer.save_model_every_n_steps=5 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=10 ++trainer.use_fsdp=false ++trainer.keep_last_n_checkpoints=-1
```

**步骤 2 — 从 checkpoint 恢复训练到 15 步:**
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_ckpt ++trainer.experiment_name=test_ckpt ++trainer.data_loading_config.max_tokens=500 ++trainer.max_steps=15 ++trainer.checkpoint_every_n_steps=5 ++trainer.save_model_every_n_steps=5 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=15 ++trainer.use_fsdp=false ++trainer.keep_last_n_checkpoints=-1
```

**验证**:
- 步骤 1 后检查: `ls checkpoints/test_ckpt/checkpoints/`，应看到 `step_5/` 和 `step_10/`
- 步骤 2 日志中应出现 `"Attempting to load last checkpoint"` 和 `"restoring training from step 10"`
- 步骤 2 后应出现 `step_15/` 目录

**清理**: `rm -rf checkpoints/test_ckpt`

---

## 测试 2: WandB 集成

**目的**: 验证 WandB 能正常初始化、记录 metrics、生成 wandb 文件。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_wandb ++trainer.experiment_name=test_wandb ++trainer.wandb_project=lcm_unit_test ++trainer.data_loading_config.max_tokens=500 ++trainer.max_steps=5 ++trainer.checkpoint_every_n_steps=5 ++trainer.save_model_every_n_steps=5 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=5 ++trainer.use_fsdp=false
```

**验证**:
- 日志中应出现 WandB 初始化信息（如 `wandb: Run data is saved locally in...`）
- 检查 `ls checkpoints/test_wandb/wandb/`，应包含 `wandb-run-*` 文件
- 打开 WandB Dashboard (`https://wandb.ai`) 查看 `lcm_unit_test` 项目中的 run
- 确认 `Train/Loss`、`Train/Learning Rate` 等 metrics 曲线存在

**清理**: `rm -rf checkpoints/test_wandb`

---

## 测试 3: FineWeb Toy Data Loading + 预处理

**目的**: 验证 FineWeb 数据能正确加载，SONAR embeddings 能正确读取，train/validation split 正常工作。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_data ++trainer.experiment_name=test_data ++trainer.data_loading_config.max_tokens=500 ++trainer.max_steps=3 ++trainer.checkpoint_every_n_steps=3 ++trainer.save_model_every_n_steps=3 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=3 ++trainer.use_fsdp=false
```

**验证**:
- 训练顺利完成 3 步（无 data loading 错误）
- 日志中能看到 batch_size、elements_per_batch 等数据统计
- 验证步骤成功执行（日志中出现 validation 相关输出）

**清理**: `rm -rf checkpoints/test_data`

---

## 测试 4: Pretrain 模式正常运作

**目的**: 验证完整的 pretrain pipeline（包括 forward/backward、optimizer step、metrics logging、checkpoint、validation）都能正常运行。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_pretrain ++trainer.experiment_name=test_pretrain ++trainer.data_loading_config.max_tokens=500 ++trainer.max_steps=10 ++trainer.checkpoint_every_n_steps=5 ++trainer.save_model_every_n_steps=5 ++trainer.publish_metrics_every_n_steps=2 ++trainer.validate_every_n_steps=5 ++trainer.use_fsdp=false
```

**验证**:
- 训练完成全部 10 步
- 日志中每 2 步输出 Train metrics（loss、lr、grad_norm 等）
- 第 5 步和第 10 步执行 validation
- 有 checkpoint 保存（`checkpoints/test_pretrain/checkpoints/step_5/` 和 `step_10/`）
- loss 值有限且在合理范围内（不是 NaN/Inf）

**清理**: `rm -rf checkpoints/test_pretrain`

---

## 测试 5: 断点恢复（模拟中断）

**目的**: 模拟训练中断后，验证从 checkpoint 恢复训练的完整性（包括 model、optimizer、dataloader 状态恢复，以及 WandB 继续记录）。

**步骤 1 — 训练 8 步，在 step 4 保存 checkpoint，然后训练完成:**
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_resume ++trainer.experiment_name=test_resume ++trainer.wandb_project=lcm_unit_test ++trainer.data_loading_config.max_tokens=500 ++trainer.max_steps=8 ++trainer.checkpoint_every_n_steps=4 ++trainer.save_model_every_n_steps=4 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=8 ++trainer.use_fsdp=false ++trainer.keep_last_n_checkpoints=-1
```

**步骤 2 — 模拟断点恢复：以 max_steps=20 重启（自动从 step 8 恢复）:**
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_resume ++trainer.experiment_name=test_resume ++trainer.wandb_project=lcm_unit_test ++trainer.data_loading_config.max_tokens=500 ++trainer.max_steps=20 ++trainer.checkpoint_every_n_steps=4 ++trainer.save_model_every_n_steps=4 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=20 ++trainer.use_fsdp=false ++trainer.keep_last_n_checkpoints=-1
```

**验证**:
- 步骤 2 日志中出现: `"Attempting to load last checkpoint"` → `"restoring training from step 8"`
- 训练从 step 9 继续到 step 20
- WandB 中 metrics 曲线应该连续（step 1-8 和 step 9-20 无间断）
- 检查 checkpoint 目录: `ls checkpoints/test_resume/checkpoints/`

**清理**: `rm -rf checkpoints/test_resume`

---

## 测试 6: LR Decay（decay_from_pretrain 模式）

**目的**: 验证 decay_from_pretrain 模式能正确加载 pretrain checkpoint，构建 decay-only LR schedule (stage_ratio=(0.0, 0.0, 1.0))，并正确进行 LR 衰减。

**步骤 1 — 先运行 20 步 pretrain 作为基础:**
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_decay ++trainer.experiment_name=test_decay_pretrain ++trainer.data_loading_config.max_tokens=500 ++trainer.max_steps=20 ++trainer.checkpoint_every_n_steps=10 ++trainer.save_model_every_n_steps=10 ++trainer.publish_metrics_every_n_steps=5 ++trainer.validate_every_n_steps=20 ++trainer.use_fsdp=false
```

**步骤 2 — 从 pretrain checkpoint 启动 decay 模式:**
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_decay ++trainer.experiment_name=test_decay_phase ++trainer.training_mode=decay_from_pretrain ++trainer.pretrain_total_steps=20 ++trainer.decay_ratio=0.5 ++trainer.num_lr_warmup_steps=0 ++trainer.data_loading_config.max_tokens=500 ++trainer.checkpoint_every_n_steps=5 ++trainer.save_model_every_n_steps=5 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=20 ++trainer.use_fsdp=false
```

**验证**:
- 步骤 2 日志中出现: `"decay_from_pretrain mode:"` 和 `"stage_ratio=(0.0, 0.0, 1.0)"`
- 日志显示 `"Loading pretrain checkpoint for decay_from_pretrain mode."`
- LR 值在每步的 metrics 输出中应从 `lr=0.0004` 逐步衰减到 `final_lr=1e-5`
- 如果设置了 `wandb_project`，可以在 WandB 上观察 LR 曲线是单调下降的

**清理**: `rm -rf checkpoints/test_decay`

---

## 一键清理所有测试输出

```bash
rm -rf checkpoints/test_ckpt checkpoints/test_wandb checkpoints/test_data checkpoints/test_pretrain checkpoints/test_resume checkpoints/test_decay
```

## 环境变量设置（可选）

如果遇到磁盘空间或缓存问题，在运行测试前设置:

```bash
export UV_CACHE_DIR=/work/hdd/bfaq/jlyu3/lcm/uv_cache
export HF_HOME=/work/hdd/bfaq/jlyu3/lcm/hf_cache
export TMPDIR=/work/hdd/bfaq/jlyu3/lcm/tmp
```
