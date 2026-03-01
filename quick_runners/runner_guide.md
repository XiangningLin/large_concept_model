# LCM 完整训练 Pipeline 运行指南

本文档提供从数据预处理到预训练到 decay 的完整 CLI 命令。采用 **bash runner + sbatch_bash_runner** 风格：每个任务有对应的 bash 脚本，可直接运行或通过 sbatch 提交。

> **约定**：
>
> - 项目根目录：`/projects/bfaq/jlyu3/large_concept_model`
> - 数据存放：`/work/hdd/bfaq/jlyu3/lcm/`
> - 使用 `uv run` 运行（或先 `source .venv/bin/activate`）
> - 模型 size 用 `N` 表示（如 60M、130M、250M、370M、780M、1.6B），recipe 为 `mse_60M`、`mse_130M` 等
> - 一个 SONAR 句子 ≈ 18.5285 subword tokens
> - 所有 recipe 默认 `batch_size=32`（per-GPU），`max_tokens=null`。8 GPU 时 global batch = 256。

### 执行方式：salloc vs sbatch


| 方式         | 说明                                                          | 适用场景           |
| ---------- | ----------------------------------------------------------- | -------------- |
| **salloc** | 先分配节点获得交互 shell，再在节点上执行 bash runner                         | 调试、短任务、需实时查看输出 |
| **sbatch** | `sbatch sbatch_runners/sbatch_bash_runner.sh <runner>` 提交作业 | 长任务、批量提交、无需交互  |


**通用 sbatch 用法**：

```bash
# 从项目根目录提交（默认 8 GPU）
sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain.sh

# 1-GPU 任务需覆盖资源
sbatch --gres=gpu:1 --mem=128G sbatch_runners/sbatch_bash_runner.sh ./quick_runners/packing/pack_parquet.sh
```

---

## 0. 环境变量（推荐设置）

```bash
export HF_TOKEN="${HF_TOKEN:-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}"
export UV_CACHE_DIR=/work/hdd/bfaq/jlyu3/lcm/uv_cache
export HF_HOME=/work/hdd/bfaq/jlyu3/lcm/hf_cache
export TMPDIR=/work/hdd/bfaq/jlyu3/lcm/tmp
```

---

## 1. 数据预处理（FineWeb-Edu-20B → Parquet + SONAR）

**脚本**：`prep_fineweb_streaming_multigpu.sh`（流式 + islice 多卡）

**数据来源**：`LGVamper/fineweb-edu-20B`（私有数据集，需 `export HF_TOKEN=xxx`）

**用户必填环境变量**：


| 变量            | 说明                                  |
| ------------- | ----------------------------------- |
| `OUTPUT_DIR`  | 输出目录（target_path）                   |
| `START_INDEX` | 起始样本索引                              |
| `NUM_SAMPLES` | 每 rank 处理样本数（总样本 ≈ 8 × NUM_SAMPLES） |


### 1a. bash 直接运行（8 卡）

```bash
# 1. 分配 8 卡节点
salloc --account=bfaq-delta-gpu --partition=gpuA100x8 --nodes=1 --gres=gpu:8 --mem=256G --time=24:00:00

# 2. 在节点上执行（必填 OUTPUT_DIR、START_INDEX、NUM_SAMPLES）
export HF_TOKEN=xxx  # 私有数据集必需
OUTPUT_DIR=/work/hdd/bfaq/jlyu3/lcm/preprocessed_data \
START_INDEX=0 \
NUM_SAMPLES=1000000 \
./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh
```

### 1b. sbatch 提交（8 卡）

```bash
export HF_TOKEN=xxx
export OUTPUT_DIR=/work/hdd/bfaq/jlyu3/lcm/preprocessed_data
export START_INDEX=0
export NUM_SAMPLES=1000000
sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh
```

### 1c. 输出与衔接

输出：`OUTPUT_DIR/rank_0/` … `rank_7/`。训练 dataloader 可直接读取；若需 packing，将此目录作为 `pack_parquet.py` 的 `--source_dir`。

---

## 2. Packing（短文档合并为固定长度行）

🖥️ **salloc** 或 📤 **sbatch** 均可。Packing 仅需 1 GPU：

```bash
# salloc 1 卡后执行
./quick_runners/packing/pack_parquet.sh

# sbatch
sbatch --gres=gpu:1 --mem=128G sbatch_runners/sbatch_bash_runner.sh ./quick_runners/packing/pack_parquet.sh
```

> 环境变量：`SOURCE_DIR`、`OUTPUT_DIR`、`MAX_SEQ_LEN`（默认 128）。在 `lcm/datacards/datacards.yaml` 中更新 `fine_web_edu_packed` 的 `parquet_path.local` 指向 packed 输出目录。

---

## 3. LR Sweep（学习率搜索）

🖥️ **salloc** 8 卡节点或 📤 **sbatch** 8 卡作业后执行：

```bash
# salloc 8 卡后执行
RECIPE=mse_370M LR_VALUES="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3" \
SWEEP_STEPS=500 WARMUP_STEPS=20 EVAL_STEPS=10 \
CUDA_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
./quick_runners/train/lr_sweep_pretrain.sh

# sbatch（先 export 环境变量，sbatch 会继承）
export RECIPE=mse_370M LR_VALUES="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3"
export SWEEP_STEPS=500 WARMUP_STEPS=20 EVAL_STEPS=10
export CUDA_DEVICES=0,1,2,3,4,5,6,7 NPROC=8
sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/lr_sweep_pretrain.sh
```

输出：`results/lr_sweep/lr_sweep_results.csv`（`peak_lr,avg_val_loss,tail_std`）。

> **高 LR（如 1e-2）fp16 溢出**：recipe 默认 fp16，高 LR 易触发梯度溢出导致重试变慢。可加 `FP32_LR_VALUES="1e-2 1e-1"` 仅对高 LR 用 fp32，或 `USE_FP32=1` 全程 fp32。

---

## 4. 预训练（Pretrain，8 卡标准）

### 计算 max_steps

```
tokens_per_sentence ≈ 18.5285
sentences_per_step  = batch_size × gradient_accumulation × num_gpus
tokens_per_step     = sentences_per_step × 18.5285
max_steps           = total_tokens / tokens_per_step
```

例如：20B tokens，batch_size=32，gradient_accumulation=1，8 GPU：

```
sentences_per_step = 32 × 1 × 8 = 256
tokens_per_step    = 256 × 18.5285 ≈ 4,743
max_steps          = 20,000,000,000 / 4,743 ≈ 4,217,000
```

### Pretrain 配置约定（SentenceSSM 风格）

- 设总 horizon 为 S 步
- **Pretrain** 跑 `0.9 × S` 步（`max_steps = 0.9S`，`lr_stage_ratios=[0.1, 0.9, 0.0]` → 无 decay）
- **Decay** 跑剩余 `0.1 × S` 步（`max_steps = S`，`decay_ratio = 0.1`，见第 5 节）

### Checkpoint milestones（照搬 SentenceSSM 的 scaling law 观察点）

```yaml
checkpoint_milestones: [225000000, 318600000, 450000000, 636300000, 900000000, 1272600000, 1800000000, 2545200000, 3600000000, 5091300000, 7200000000, 10182600000, 14400000000]
```

### 启动预训练（8 卡）

以 370M 模型为例（S = 4752，pretrain 跑 0.9S = 4277 步）：

```bash
# salloc 8 卡后执行
./quick_runners/train/pretrain.sh

# sbatch
sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain.sh
```

> **环境变量**：`RECIPE`（默认 mse_370M）、`OUTPUT_DIR`、`EXPERIMENT_NAME`、`MAX_STEPS`、`WANDB_PROJECT`。
> **替换 RECIPE**：`RECIPE=mse_60M ./quick_runners/train/pretrain.sh` 或 `RECIPE=mse`（1.6B）。
> **使用 packed 数据**：追加 `'++trainer.training_data.0.name="fine_web_edu_packed=train"' ++trainer.training_data.0.source_suffix_text=null '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' ++trainer.validation_data.0.source_suffix_text=null`。

### 4.5. 预训练中断恢复

```bash
RESUME_FROM=/work/hdd/bfaq/jlyu3/lcm/checkpoints/lcm_370m_pretrain/milestones/milestone_tokens_3600000000 \
WANDB_RUN_ID=<FILL_ORIGINAL_RUN_ID> \
./quick_runners/train/pretrain_resume.sh

# sbatch（先 export 环境变量）
export RESUME_FROM=/work/hdd/bfaq/jlyu3/lcm/checkpoints/lcm_370m_pretrain/milestones/milestone_tokens_3600000000
export WANDB_RUN_ID=<FILL_ORIGINAL_RUN_ID>
sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain_resume.sh
```

---

## 5. Decay 阶段（decay_from_pretrain，8 卡标准）

从 pretrain 的 milestone checkpoint 开始，LR scheduler 使用完整 S 步 horizon 的 WSD 曲线（warmup+stable+decay）。

```bash
# salloc 8 卡后执行（默认从 milestone_tokens_14400000000 恢复）
./quick_runners/train/decay.sh

# sbatch
sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/decay.sh
```

> **环境变量**：`RESUME_FROM`（默认最后一个 pretrain milestone）、`WANDB_RUN_ID`、`OUTPUT_DIR`、`MAX_STEPS`、`DECAY_RATIO`（默认 0.1）。

---

## 可用模型 Recipes


| Recipe     | model_arch      | 参数量   |
| ---------- | --------------- | ----- |
| `mse_60M`  | `base_lcm_60M`  | ~62M  |
| `mse_130M` | `base_lcm_130M` | ~130M |
| `mse_250M` | `base_lcm_250M` | ~250M |
| `mse_370M` | `base_lcm_370M` | ~370M |
| `mse_780M` | `base_lcm_780M` | ~780M |
| `mse`      | `base_lcm_1_6B` | ~1.6B |


---

## 执行方式速查表（8 卡标准）


| 步骤                | salloc                                                                                                                      | sbatch                                                                                             | Runner                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **1. 预处理（8 卡流式）** | salloc 8 卡 → `OUTPUT_DIR=... START_INDEX=... NUM_SAMPLES=... ./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh` | 同上，先 export 三变量后 sbatch                                                                            | prep_fineweb_streaming_multigpu.sh |
| **2. Packing**    | salloc 1 卡 → `./quick_runners/packing/pack_parquet.sh`                                                                      | `sbatch --gres=gpu:1 sbatch_runners/sbatch_bash_runner.sh ./quick_runners/packing/pack_parquet.sh` | pack_parquet.sh                    |
| **3. LR Sweep**   | salloc 8 卡 → `RECIPE=... CUDA_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 ./quick_runners/train/lr_sweep_pretrain.sh`                   | 同上，sbatch 包装                                                                                       | lr_sweep_pretrain.sh               |
| **4. Pretrain**   | salloc 8 卡 → `./quick_runners/train/pretrain.sh`                                                                            | `sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain.sh`                    | pretrain.sh                        |
| **5. Decay**      | salloc 8 卡 → `./quick_runners/train/decay.sh`                                                                               | `sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/decay.sh`                       | decay.sh                           |


> **分区说明**：若集群有 8 卡节点分区（如 `gpuA100x8`），优先使用；否则用 4 卡分区（如 `gpuA100x4`）并相应调整 `--nproc-per-node`。sbatch 默认在 `sbatch_runners/sbatch_bash_runner.sh` 中配置。

