# LCM 功能模块测试指南

所有测试使用 **mse_60M** 模型（~62M参数），按**正确依赖顺序**执行：1.预处理 → 2.Pack+Merge → 3.LR Sweep → 4.预训练/断点恢复/Decay。**测试 3 和 4 自动使用前序步骤生成的 packed 数据**（`output/fine_web_test_packed`）。

> **前置条件**: 确保 `datacards.yaml` 中 `fine_web_edu_packed.parquet_path.s3` 指向 `output/fine_web_test_packed`。若使用 `uv`，所有命令需加 `uv run` 前缀以使用 `.venv` 环境；也可先执行 `source .venv/bin/activate` 激活环境后直接运行。
>
> **batch_size 说明**: 所有 recipe 已默认 `batch_size=32, max_tokens=null`。toy 测试中通过 `++trainer.data_loading_config.batch_size=4` 覆盖为小 batch 以加速。
>
> **多卡路径**: 所有训练测试均使用 `torchrun` 启动（多卡 capable），单卡时 `--nproc-per-node=1`，多卡时增大即可（如 `--nproc-per-node=8`）。

---

## 测试 1: 预处理

**目的**: 从零开始预处理小数据集（流式 islice 方案），验证 `prepare_fine_web.py` 流式模式。**支持断点续跑**：启用 `--checkpoint_interval` 后，中断重跑会从 checkpoint 恢复。输出供测试 2 使用。

### 1.1 流式预处理（单卡，50 条，启用 checkpoint）

```bash
export HF_TOKEN="${HF_TOKEN:-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}"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_fine_web.py \
  --num_samples=50 --output_dir=output/fine_web_test_pack_src \
  --rank=0 --num_shards=1 --checkpoint_interval=10
```

**验证**: 输出 `output/fine_web_test_pack_src/data.parquet`，约 50 条；`checkpoints/` 下有 `data_checkpoint_*.parquet`。

### 1.2 流式预处理断点续跑（可选）

**步骤 A — 首次运行约一半后中断（如 Ctrl+C）：**

```bash
rm -rf output/fine_web_test_pack_resume
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_fine_web.py \
  --num_samples=50 --output_dir=output/fine_web_test_pack_resume \
  --rank=0 --num_shards=1 --checkpoint_interval=10
# 运行到约 25 条时 Ctrl+C 中断
```

**步骤 B — 重跑同一命令，应从 checkpoint 续跑：**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_fine_web.py \
  --num_samples=50 --output_dir=output/fine_web_test_pack_resume \
  --rank=0 --num_shards=1 --checkpoint_interval=10
```

**验证**: 步骤 B 日志出现「检测到检查点文件，尝试从检查点恢复」；最终 `data.parquet` 约 50 条。

> 若跳过 1.2，可直接用 1.1 的输出做测试 2。测试 2 的 `--source_dir` 可用 `output/fine_web_test_pack_src` 或 `output/fine_web_test_pack_resume`。

---

## 测试 2: Pack + Merge Chunks

**目的**: 对测试 1 的输出做 packing，并验证多 chunk 合并脚本。主流程输出 `output/fine_web_test_packed`，供测试 3、4 使用。

### 2.1 Packing（主流程，单 chunk）

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/pack_parquet.py \
  --source_dir output/fine_web_test_pack_src \
  --output_dir output/fine_web_test_packed \
  --max_seq_len 128
```

**验证**: 打印 packing 统计，无报错；输出 `output/fine_web_test_packed/split=train/` 等。

### 2.2 Packed Data Merge（多 chunk 合并，可选）

**步骤 1 — 制造两个 toy chunk:**

```bash
export HF_TOKEN="${HF_TOKEN:-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}"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_fine_web.py --num_samples=30 --output_dir=output/merge_test/chunk_0
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_fine_web.py --num_samples=30 --start_index=30 --output_dir=output/merge_test/chunk_1
```

**步骤 2 — 分别 pack:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/pack_parquet.py --source_dir=output/merge_test/chunk_0 --output_dir=output/merge_test_packed/chunk_0 --max_seq_len=128
CUDA_VISIBLE_DEVICES=0 uv run python scripts/pack_parquet.py --source_dir=output/merge_test/chunk_1 --output_dir=output/merge_test_packed/chunk_1 --max_seq_len=128
```

**步骤 3 — 合并:**

```bash
bash scripts/merge_packed_chunks.sh --mode chunk --source output/merge_test_packed --output output/merge_test_packed_merged
```

**验证**: 输出目录包含 `split=train/data_chunk_0.parquet`、`split=train/data_chunk_1.parquet`（validation 同理）；脚本打印 row count 统计。

> **注意**: 测试 3、4 使用 2.1 的输出 `output/fine_web_test_packed`。若改用 merged 输出，需在 `datacards.yaml` 中更新 `fine_web_edu_packed.parquet_path.s3`。

### 2.3 Packing 单元测试（无需 GPU）

```bash
uv run python -m pytest mytests/test_pack_parquet.py -v
```

**验证**: 全部 14 个测试用例通过（包括 split、greedy_pack、end-to-end parquet I/O）。

---

## 测试 3: LR Sweep（学习率搜索）

**目的**: 验证 LR sweep 功能：短训练 + tail-only eval + parseable output。**使用测试 1、2 生成的 packed 数据**（`output/fine_web_test_packed`）。

**步骤 1 — 单次 lr_sweep_mode 验证（50 步，tail_ratio=0.2，eval 每 10 步）:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_sweep ++trainer.experiment_name=test_sweep ++trainer.lr_sweep_mode=true ++trainer.tail_ratio=0.2 ++trainer.max_steps=50 ++trainer.num_lr_warmup_steps=5 ++trainer.lr=0.001 ++trainer.lr_schedule=noop ++trainer.validate_every_n_steps=10 ++trainer.checkpoint_every_n_steps=999999 ++trainer.save_model_every_n_steps=999999 ++trainer.publish_metrics_every_n_steps=10 ++trainer.use_fsdp=false ++trainer.data_loading_config.batch_size=4 '++trainer.checkpoint_milestones=[]' '++trainer.training_data.0.name="fine_web_edu_packed=train"' ++trainer.training_data.0.source_suffix_text=null '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' ++trainer.validation_data.0.source_suffix_text=null
```

**步骤 2 — 用 sweep 脚本跑 3 个 LR 值（每个 50 步）:**

```bash
LR_VALUES="1e-4 1e-3 1e-2" SWEEP_STEPS=50 WARMUP_STEPS=5 EVAL_STEPS=10 ./quick_runners/train/lr_sweep_pretrain.sh ++trainer.data_loading_config.batch_size=4 '++trainer.training_data.0.name="fine_web_edu_packed=train"' ++trainer.training_data.0.source_suffix_text=null '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' ++trainer.validation_data.0.source_suffix_text=null
```

**验证**:

- 步骤 1: 训练完成后 stdout 中出现 `FINAL LOSS: tail_avg_val_loss=..., tail_std=..., tail_n=...`（tail_n 应为 1，因为 tail 区间 step 40-50 中 eval 在 step 50 触发 1 次）
- 步骤 1: 日志中出现 `"LR sweep mode: tail eval starts at step 40"`
- 步骤 2: 生成 `results/lr_sweep/lr_sweep_results.csv`，包含 3 行数据（header + 3 个 LR 值）
- 步骤 2: 终端输出每个 LR 对应的 avg_val_loss 和 tail_std

**清理**: `rm -rf checkpoints/test_sweep checkpoints/lr_sweep_tmp results/lr_sweep`

---

## 测试 4: 预训练、断点恢复、Decay

**目的**: 验证 pretrain pipeline、checkpoint 保存与恢复、WandB 集成、decay_from_pretrain 模式。**使用测试 1、2 生成的 packed 数据**（`output/fine_web_test_packed`）。

### 4.1 Packed 数据短训练（5 步，验证端到端）

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_packed ++trainer.experiment_name=test_packed ++trainer.data_loading_config.batch_size=4 ++trainer.max_steps=5 '++trainer.checkpoint_milestones=[1000,2500]' ++trainer.checkpoint_every_n_steps=999999 ++trainer.save_model_every_n_steps=999999 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=5 ++trainer.use_fsdp=false '++trainer.training_data.0.name="fine_web_edu_packed=train"' ++trainer.training_data.0.source_suffix_text=null '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' ++trainer.validation_data.0.source_suffix_text=null
```

**验证**: 训练顺利完成 5 步，loss 值有限。

### 4.2 Pretrain + Checkpoint + Resume + WandB

> **断点恢复**：默认 `resume_from=null` 时训练从头开始；显式设置 `resume_from` 时加载 checkpoint 继续。支持 milestone 路径、`resume_step`、HuggingFace Hub。

**步骤 1 — 训练 15 步，按 token 里程碑保存 checkpoint，启用 WandB:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_pretrain ++trainer.experiment_name=test_pretrain ++trainer.wandb_project=lcm_unit_test ++trainer.data_loading_config.batch_size=4 ++trainer.max_steps=15 '++trainer.checkpoint_milestones=[1000,2500,5000]' ++trainer.checkpoint_every_n_steps=999999 ++trainer.save_model_every_n_steps=999999 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=15 ++trainer.use_fsdp=false ++trainer.keep_last_n_checkpoints=-1 '++trainer.training_data.0.name="fine_web_edu_packed=train"' ++trainer.training_data.0.source_suffix_text=null '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' ++trainer.validation_data.0.source_suffix_text=null
```

**步骤 2 — 从 milestone_tokens_5000 恢复训练到 30 步:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_pretrain ++trainer.experiment_name=test_pretrain ++trainer.resume_from=checkpoints/test_pretrain/milestones/milestone_tokens_5000 ++trainer.wandb_project=lcm_unit_test ++trainer.wandb_run_id=<FILL_ORIGINAL_RUN_ID> ++trainer.data_loading_config.batch_size=4 ++trainer.max_steps=30 '++trainer.checkpoint_milestones=[1000,2500,5000,10000,20000]' ++trainer.checkpoint_every_n_steps=999999 ++trainer.save_model_every_n_steps=999999 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=30 ++trainer.use_fsdp=false ++trainer.keep_last_n_checkpoints=-1 '++trainer.training_data.0.name="fine_web_edu_packed=train"' ++trainer.training_data.0.source_suffix_text=null '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' ++trainer.validation_data.0.source_suffix_text=null
```

> `wandb_run_id`: 从步骤 1 的 WandB Dashboard 获取 run ID 填入，或删除此参数创建新 run。

**验证**:

- 步骤 1: 训练完成 15 步，`ls checkpoints/test_pretrain/milestones/` 有 `milestone_tokens_1000/`、`milestone_tokens_2500/`、`milestone_tokens_5000/`
- 步骤 1: 日志有 WandB 初始化、Train metrics、validation
- 步骤 2: 日志出现 `"Loading checkpoint from resume_from path"` 或 `"restoring training from step N"`
- 步骤 2: 训练继续到 step 30，WandB 曲线连续

### 4.3 LR Decay（decay_from_pretrain 模式）

**目的**: 验证 decay_from_pretrain 模式能正确加载 pretrain milestone checkpoint，构建完整 S 步 horizon 的 WSD LR schedule，并正确进行 LR 衰减。Pretrain 使用 warmup+stable only（lr_stage_ratios=[0.1, 0.9, 0.0]），milestone 可落在任意步。

**配置约定**（SentenceSSM 风格）：pretrain 用 `max_steps=0.9S`；decay 用 `max_steps=S`、`decay_ratio=0.1`。例：S=60 时 pretrain 跑 54 步，decay 跑 6 步。

> **稳定续训**: 如果 checkpoint 的步数 < 0.9S，scheduler 会先以 peak_lr 继续 stable 阶段到 0.9S，再开始 decay。

**步骤 1 — 先运行 pretrain（仅 warmup+stable，无 decay），按 token 里程碑保存 checkpoint:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_decay ++trainer.experiment_name=test_decay_pretrain ++trainer.data_loading_config.batch_size=4 ++trainer.max_steps=54 '++trainer.lr_stage_ratios=[0.1,0.9,0.0]' '++trainer.checkpoint_milestones=[5000,10000]' ++trainer.checkpoint_every_n_steps=999999 ++trainer.save_model_every_n_steps=999999 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=54 ++trainer.use_fsdp=false '++trainer.training_data.0.name="fine_web_edu_packed=train"' ++trainer.training_data.0.source_suffix_text=null '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' ++trainer.validation_data.0.source_suffix_text=null
```

**步骤 2 — 从 pretrain milestone checkpoint 启动 decay 模式（max_steps=S，decay_ratio=0.1）:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.train launcher=standalone +pretrain=mse_60M ++trainer.output_dir=checkpoints/test_decay ++trainer.experiment_name=test_decay_phase ++trainer.training_mode=decay_from_pretrain ++trainer.resume_from=checkpoints/test_decay/milestones/milestone_tokens_10000 ++trainer.max_steps=60 ++trainer.decay_ratio=0.1 ++trainer.data_loading_config.batch_size=4 '++trainer.checkpoint_milestones=[15000,20000]' ++trainer.checkpoint_every_n_steps=999999 ++trainer.save_model_every_n_steps=999999 ++trainer.publish_metrics_every_n_steps=1 ++trainer.validate_every_n_steps=25 ++trainer.use_fsdp=false '++trainer.training_data.0.name="fine_web_edu_packed=train"' ++trainer.training_data.0.source_suffix_text=null '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' ++trainer.validation_data.0.source_suffix_text=null
```

> S=60：pretrain 跑 54 步 (0.9S)，decay 从 checkpoint 跑到 step 60（6 步 decay）。
> 如果 checkpoint 步数 < 54，训练会先在 stable LR 阶段继续直到 step 54，再开始 decay。

**验证**:

- 步骤 2 日志中出现: `"decay_from_pretrain mode:"` 和 `"stage_ratio=(..., ..., 0.1) (full-horizon WSD)"`
- 日志显示 `"Loading pretrain checkpoint for decay_from_pretrain mode."`
- 日志显示 `"LR scheduler advanced to step N"`
- LR 值在 decay 阶段应从 `lr=0.0004` 逐步衰减到 `final_lr=1e-5`
- 如果设置了 `wandb_project`，可以在 WandB 上观察 LR 曲线

**清理**: `rm -rf checkpoints/test_pretrain checkpoints/test_decay checkpoints/test_packed`

---

## 一键清理所有测试输出

```bash
rm -rf checkpoints/test_pretrain checkpoints/test_decay checkpoints/test_packed checkpoints/test_sweep checkpoints/lr_sweep_tmp output/fine_web_test_pack_src output/fine_web_test_pack_resume output/fine_web_test_packed output/fine_web_test_parallel output/fine_web_test_resume output/fine_web_test_resume_cache output/merge_test output/merge_test_packed output/merge_test_packed_merged results/lr_sweep
```

## 环境变量设置（可选）

如果遇到磁盘空间或缓存问题，在运行测试前设置:

```bash
export UV_CACHE_DIR=/work/hdd/bfaq/jlyu3/lcm/uv_cache
export HF_HOME=/work/hdd/bfaq/jlyu3/lcm/hf_cache
export TMPDIR=/work/hdd/bfaq/jlyu3/lcm/tmp
```

