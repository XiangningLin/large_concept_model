# LCM Code Run Guide

本指南提供数据预处理、预训练、断点恢复和 decay-from-pretrain 的 CLI 命令示例。

---

## 1. FineWeb 数据预处理（支持 start index）

```bash
python preprocess_fineweb.py --lang eng --output_dir /work/hdd/bfaq/jlyu3/lcm/preprocessed_data/chunk_1 --num_gpus 2 --start_index 100 --num_texts 500000
```

> `--start_index` 控制从 FineWeb 数据集的第几条样本开始处理，`--num_texts` 控制处理总量。

---

## 2.1 普通 Pretrain

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m lcm.train launcher=standalone +pretrain=mse_780M ++trainer.output_dir=checkpoints/lcm_pretrain ++trainer.max_steps=100000 ++trainer.data_loading_config.max_tokens=6000 ++trainer.lr_schedule=wsd '++trainer.training_data.0.name="pretraining_data=train"' '++trainer.validation_data.0.name="pretraining_data=validation"'
```

---

## 2.2 断点恢复 Pretrain（从 HuggingFace Hub 下载 checkpoint）

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m lcm.train launcher=standalone +pretrain=mse_780M ++trainer.output_dir=checkpoints/lcm_pretrain ++trainer.max_steps=100000 ++trainer.data_loading_config.max_tokens=6000 ++trainer.hf_repo_id=LGVamper/Sentence-SSM ++trainer.hf_token=hf_xxx ++trainer.hf_checkpoint_subfolder=lcm_780M_pretrain ++trainer.hf_checkpoint_filename=step_50000 ++trainer.wandb_run_id=abc123def '++trainer.training_data.0.name="pretraining_data=train"' '++trainer.validation_data.0.name="pretraining_data=validation"'
```

> 断点恢复时，设置 `wandb_run_id` 为 pretrain 启动时的 WandB run ID 可续接同一曲线。
> 本地有 checkpoint 则直接恢复（包括数据进度和 RNG 状态）；否则从 HF Hub 下载后恢复。

---

## 2.3 Decay-from-Pretrain

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m lcm.train launcher=standalone +decay_from_pretrain=mse ++trainer.output_dir=checkpoints/lcm_decay ++trainer.pretrain_total_steps=100000 ++trainer.decay_ratio=0.1 ++trainer.num_lr_warmup_steps=0 ++trainer.hf_repo_id=LGVamper/Sentence-SSM ++trainer.hf_token=hf_xxx ++trainer.hf_checkpoint_subfolder=lcm_780M_pretrain ++trainer.hf_checkpoint_filename=step_100000 ++trainer.hf_checkpoint_save_subfolder=lcm_780M_decay ++trainer.wandb_run_id=abc123def '++trainer.training_data.0.name="pretraining_data=train"' '++trainer.validation_data.0.name="pretraining_data=validation"'
```

> `max_steps` 自动计算：`pretrain_total_steps × decay_ratio / (1 - decay_ratio) = 100000 × 0.1 / 0.9 ≈ 11111`
> Decay 继承 pretrain 的 step_nr（如 100001 开始）、数据进度、RNG 状态、optimizer 状态。
> 只有 lr_scheduler（使用全新 decay-only WSD）和 milestones（使用 recipe 中新配置的）是全新的。
> WandB 自动命名为 `lcm_decay_from_pretrain_{timestamp}`，或设 `wandb_run_id` 续接 pretrain run。

---

## 配置参数说明

| 参数 | 说明 |
|------|------|
| `training_data.0.name` | Hydra list 索引语法，`.0` = 第一个数据集 |
| `wandb_run_id` | WandB run ID，设为 pretrain 的 run ID 可续接曲线 |
| `pretrain_total_steps` | Pretrain 阶段实际运行步数（不含 decay） |
| `decay_ratio` | Decay 阶段占总预算 S 的比例（S = pretrain + decay） |
| `num_lr_warmup_steps` | WSD 调度器 warmup 步数，decay 模式下可配置 |
| `hf_checkpoint_filename` | 指定加载的 checkpoint 文件名 |
| `hf_checkpoint_save_subfolder` | 保存 checkpoint 到 HF 的不同子目录 |
