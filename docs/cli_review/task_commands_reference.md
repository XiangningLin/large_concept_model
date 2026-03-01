# LCM 三大任务场景：命令形式与配置参数完整参考

本文档汇总**数据预处理**、**LR Sweep**、**Scaling Law（Pretrain + Decay）** 三大任务场景下，所有可用的命令形式及配置参数。默认环境为 **8 卡**，支持 **bash 直接运行** 和 **sbatch 提交** 两种形式。

---

## 约定

| 约定项 | 说明 |
|--------|------|
| 项目根目录 | `/projects/bfaq/jlyu3/large_concept_model` |
| 运行方式 | `uv run` 或 `source .venv/bin/activate` 后直接运行 |
| 8 卡环境 | `NPROC=8`，`CUDA_DEVICES=0,1,2,3,4,5,6,7` |
| sbatch 默认 | 8 GPU、256G 内存、48h；1-GPU 任务需 `--gres=gpu:1 --mem=128G` 覆盖 |

---

## 执行方式速查

| 方式 | 说明 | 适用 |
|------|------|------|
| **bash** | salloc 分配节点后，在节点上直接执行 runner 脚本 | 调试、短任务、需实时输出 |
| **sbatch** | `sbatch sbatch_runners/sbatch_bash_runner.sh <runner>` | 长任务、批量提交、无需交互 |

---

# 1. 数据预处理（FineWeb-Edu → Parquet + SONAR）

## 1.1 命令形式

### 1.1.1 多 GPU 流式预处理（8 卡标准）

**脚本**：`quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh`

| 形式 | 命令 |
|------|------|
| **bash** | `salloc` 8 卡后执行：<br>`./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh` |
| **sbatch** | `sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh` |

**说明**：通过 bash 循环启动多进程，每个进程用 `rank/num_shards` 做 islice 交错分片。输出为 `output_dir/rank_0/` … `rank_N/`。

> **私有数据集**：若使用 LGVamper/fineweb-edu-20B 等私有数据集，需设置 `export HF_TOKEN=xxx` 或通过 `--hf_token` 传入。

### 1.1.2 单 GPU 预处理（调试/小数据）

| 形式 | 命令 |
|------|------|
| **bash** | `salloc` 1 卡后：<br>`NUM_GPUS=1 ./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh` |
| **sbatch** | `sbatch --gres=gpu:1 --mem=64G sbatch_runners/sbatch_bash_runner.sh bash -c "NUM_GPUS=1 ./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh"` |

### 1.1.3 直接调用 Python 脚本（单卡）

```bash
# 单卡，指定样本数
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_fine_web.py \
  --num_samples=10000 \
  --output_dir=/work/hdd/bfaq/jlyu3/lcm/preprocessed_data \
  --checkpoint_interval=5000
```

---

## 1.2 预处理环境变量（Runner 脚本）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_DIR` | `/work/hdd/bfaq/jlyu3/lcm/preprocessed_data` | 输出目录 |
| `NUM_GPUS` | `8` | 并行 GPU 数 |
| `START_INDEX` | `0` | 起始样本索引（多卡分段用） |
| `NUM_SAMPLES` | `1000000` | 每 rank 处理样本数（总样本 ≈ NUM_GPUS × NUM_SAMPLES） |

---

## 1.3 预处理 CLI 参数（`prepare_fine_web.py`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | `output/fine_web` | 输出目录 |
| `num_samples` | int\|None | `None` | 样本数；`None` 表示全量 |
| `start_index` | int | `0` | 起始索引（多 GPU 分段） |
| `rank` | int | 环境变量 `RANK` 或 `0` | 当前进程 rank |
| `num_shards` | int | 环境变量 `WORLD_SIZE` 或 `1` | 总进程数 |
| `batch_size` | int | `10` | SONAR 编码批大小 |
| `max_sentence_length` | int | `256` | 句子最大长度 |
| `add_split_column` | bool | `True` | 是否添加 train/validation split |
| `train_ratio` | float | `0.8` | 训练集比例 |
| `seed` | int | `42` | 随机种子 |
| `checkpoint_interval` | int | `1000` | 每 N 样本保存 checkpoint；0 禁用 |
| `enable_unk_filter` | bool | `True` | 过滤含 SONAR UNK 的句子 |

---

## 1.4 Packing（短文档合并为固定长度行）

**脚本**：`quick_runners/packing/pack_parquet.sh`

| 形式 | 命令 |
|------|------|
| **bash** | salloc 1 卡后：<br>`./quick_runners/packing/pack_parquet.sh` |
| **sbatch** | `sbatch --gres=gpu:1 --mem=128G sbatch_runners/sbatch_bash_runner.sh ./quick_runners/packing/pack_parquet.sh` |

### Packing 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SOURCE_DIR` | `/work/hdd/bfaq/jlyu3/lcm/preprocessed_data` | 预处理输出目录 |
| `OUTPUT_DIR` | `/work/hdd/bfaq/jlyu3/lcm/preprocessed_data_packed` | packed 输出目录 |
| `MAX_SEQ_LEN` | `128` | 每行最大句子数 |

### Packing CLI 参数（`pack_parquet.py`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source_dir` | str | 必填 | 预处理 parquet 目录 |
| `output_dir` | str | 必填 | packed 输出目录 |
| `max_seq_len` | int | `128` | 每行最大句子数 |
| `suffix_text` | str | `"End of text."` | 文档边界标记 |
| `device` | str | `cuda:0` | SONAR 设备 |

> **数据衔接**：在 `lcm/datacards/datacards.yaml` 中更新 `fine_web_edu_packed.parquet_path.local` 指向 packed 输出；训练时用 `++trainer.training_data.0.name="fine_web_edu_packed=train"` 等覆盖。

---

# 2. LR Sweep（学习率搜索）

## 2.1 命令形式

**脚本**：`quick_runners/train/lr_sweep_pretrain.sh`

| 形式 | 命令 |
|------|------|
| **bash** | salloc 8 卡后：<br>`RECIPE=mse_370M LR_VALUES="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3" SWEEP_STEPS=500 WARMUP_STEPS=20 EVAL_STEPS=10 CUDA_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 ./quick_runners/train/lr_sweep_pretrain.sh` |
| **sbatch** | `export RECIPE=mse_370M LR_VALUES="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3" SWEEP_STEPS=500 WARMUP_STEPS=20 EVAL_STEPS=10 CUDA_DEVICES=0,1,2,3,4,5,6,7 NPROC=8`<br>`sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/lr_sweep_pretrain.sh` |

**输出**：`results/lr_sweep/lr_sweep_results.csv`（`peak_lr,avg_val_loss,tail_std`）

---

## 2.2 LR Sweep 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LR_VALUES` | `1e-5 2e-5 ... 1e-1` | 要 sweep 的 LR 列表（空格分隔） |
| `RECIPE` | `mse_60M` | Hydra recipe 名 |
| `SWEEP_STEPS` | `500` | 每个 LR 的总训练步数 |
| `WARMUP_STEPS` | `20` | LR warmup 步数 |
| `EVAL_STEPS` | `10` | 每 N 步做一次 validation |
| `TAIL_RATIO` | `0.2` | tail 占步数比例（用于 tail 起始步） |
| `RESULTS_DIR` | `./results/lr_sweep/` | CSV 输出目录 |
| `NPROC` | `1` | 每节点 GPU 数（8 卡时设为 8） |
| `CUDA_DEVICES` | `0` | `CUDA_VISIBLE_DEVICES` |
| `USE_FP32` | 未设置 | `1` 或 `true` 时全程 fp32（避免高 LR 溢出） |
| `FP32_LR_VALUES` | 未设置 | 仅对指定 LR 用 fp32，如 `"1e-2 1e-1"` |

---

## 2.3 LR Sweep 可追加的 Hydra 覆盖

通过脚本末尾 `"$@"` 传入，例如：

```bash
./quick_runners/train/lr_sweep_pretrain.sh \
  ++trainer.data_loading_config.batch_size=4 \
  '++trainer.training_data.0.name="fine_web_edu_packed=train"' \
  ++trainer.training_data.0.source_suffix_text=null \
  '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' \
  ++trainer.validation_data.0.source_suffix_text=null \
  ++trainer.wandb_project=lcm_lr_sweep
```

| 参数 | 说明 |
|------|------|
| `++trainer.training_data.0.name` | 训练数据 datacard 名 |
| `++trainer.validation_data.0.name` | 验证数据 datacard 名 |
| `++trainer.data_loading_config.batch_size` | batch size |
| `++trainer.wandb_project` | WandB 项目名 |
| `++trainer.dtype` | 由 `USE_FP32` / `FP32_LR_VALUES` 自动覆盖，也可手动 `torch.float32` |

---

# 3. Scaling Law（Pretrain + Decay）

Scaling Law 任务 = **Pretrain**（0.9S 步，无 decay）+ **Decay**（0.1S 步，从 milestone 恢复）。按 token 里程碑保存 checkpoint，用于后续 decay 与评估。

## 3.1 Pretrain 命令形式

**脚本**：`quick_runners/train/pretrain.sh`

| 形式 | 命令 |
|------|------|
| **bash** | salloc 8 卡后：<br>`./quick_runners/train/pretrain.sh` |
| **sbatch** | `sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain.sh` |

### Pretrain 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RECIPE` | `mse_370M` | 模型 recipe（mse_60M、mse_130M、mse_250M、mse_370M、mse_780M、mse） |
| `OUTPUT_DIR` | `/work/hdd/bfaq/jlyu3/lcm/checkpoints/lcm_370m_pretrain` | checkpoint 输出目录 |
| `EXPERIMENT_NAME` | `lcm_370m_pretrain` | 实验名 |
| `WANDB_PROJECT` | `lcm_pretrain` | WandB 项目 |
| `MAX_STEPS` | `4277` | 总训练步数（370M 示例：S=4752，pretrain 0.9S） |
| `NPROC` | `8` | GPU 数 |

### Pretrain 可追加 Hydra 覆盖

```bash
./quick_runners/train/pretrain.sh \
  '++trainer.training_data.0.name="fine_web_edu_packed=train"' \
  ++trainer.training_data.0.source_suffix_text=null \
  '++trainer.validation_data.0.name="fine_web_edu_packed=validation"' \
  ++trainer.validation_data.0.source_suffix_text=null
```

---

## 3.2 Pretrain 断点恢复

**脚本**：`quick_runners/train/pretrain_resume.sh`

| 形式 | 命令 |
|------|------|
| **bash** | `RESUME_FROM=/path/to/milestone WANDB_RUN_ID=<run_id> ./quick_runners/train/pretrain_resume.sh` |
| **sbatch** | `export RESUME_FROM=... WANDB_RUN_ID=...`<br>`sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain_resume.sh` |

| 变量 | 说明 |
|------|------|
| `RESUME_FROM` | **必填**，milestone checkpoint 路径 |
| `WANDB_RUN_ID` | 可选，续接 WandB 曲线 |

---

## 3.3 Decay 命令形式

**脚本**：`quick_runners/train/decay.sh`

| 形式 | 命令 |
|------|------|
| **bash** | `./quick_runners/train/decay.sh` |
| **sbatch** | `sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/decay.sh` |

### Decay 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RESUME_FROM` | `.../milestone_tokens_14400000000` | pretrain milestone 路径 |
| `RECIPE` | `mse_370M` | 与 pretrain 一致 |
| `OUTPUT_DIR` | `.../lcm_370m_decay` | decay 输出目录 |
| `EXPERIMENT_NAME` | `lcm_370m_decay` | 实验名 |
| `WANDB_PROJECT` | `lcm_pretrain` | WandB 项目 |
| `WANDB_RUN_ID` | 未设置 | 填入 pretrain run ID 可续接曲线 |
| `MAX_STEPS` | `4752` | 总 horizon S（decay 跑 0.1S） |
| `DECAY_RATIO` | `0.1` | decay 占总 horizon 比例 |
| `NPROC` | `8` | GPU 数 |

---

## 3.4 Scaling Law 相关 Trainer 配置参数

### 输出与身份

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `??` | checkpoint / log 输出目录 |
| `experiment_name` | `null` | 实验名 |
| `training_mode` | `"pretrain"` | `pretrain` 或 `decay_from_pretrain` |
| `resume_from` | `null` | 断点恢复路径 |
| `dtype` | `torch.float16` | 计算精度 |

### 学习率与调度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | `0.0004` | 峰值学习率 |
| `lr_schedule` | `wsd` | `noop` / `myle` / `cosine` / `wsd` |
| `lr_stage_ratios` | `[0.1, 0.9, 0.0]`（pretrain） | WSD 三阶段比例；pretrain 无 decay |
| `num_lr_warmup_steps` | `10000` | warmup 步数 |
| `max_steps` | recipe 指定 | 总训练步数 |
| `decay_ratio` | `0.1` | decay 阶段占比（仅 decay 模式） |

### Token 里程碑（Scaling Law 观察点）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `checkpoint_milestones` | 见 recipe | token 数列表，达到时保存 milestone |
| `tokens_per_sentence` | `18.5` | 每句平均 subword token 数 |

**SentenceSSM 风格 milestone 示例**：
```yaml
[225000000, 318600000, 450000000, 636300000, 900000000, 1272600000, 1800000000, 2545200000, 3600000000, 5091300000, 7200000000, 10182600000, 14400000000]
```

### 数据

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training_data.0.name` | `"pretraining_data=train"` | datacard 名，如 `"fine_web_edu_packed=train"` |
| `training_data.0.source_suffix_text` | `"End of text."` | packed 数据可设为 `null` |
| `validation_data.0.name` | `"pretraining_data=validation"` | 验证集 datacard |
| `data_loading_config.batch_size` | `32` | 每 GPU batch size |
| `data_loading_config.max_tokens` | `null` | 每批最大 token 数 |

### WandB

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `wandb_project` | `null` | 项目名 |
| `wandb_run_name` | `null` | run 名；null 时自动生成 |
| `wandb_run_id` | `null` | 续接已有 run（断点 / decay） |

### 验证与 checkpoint

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `validate_every_n_steps` | `10000` | 每 N 步验证 |
| `publish_metrics_every_n_steps` | `100` | 每 N 步上报指标 |
| `checkpoint_every_n_steps` | `2000` | 每 N 步保存 checkpoint |
| `save_model_every_n_steps` | `2000` | 每 N 步保存 consolidated 模型 |
| `keep_last_n_checkpoints` | `2` | 保留最新 N 个；-1 保留全部 |

### 模型 Recipe 与规模

| Recipe | model_arch | 参数量 |
|--------|------------|--------|
| `mse_60M` | base_lcm_60M | ~62M |
| `mse_130M` | base_lcm_130M | ~130M |
| `mse_250M` | base_lcm_250M | ~250M |
| `mse_370M` | base_lcm_370M | ~370M |
| `mse_780M` | base_lcm_780M | ~780M |
| `mse` | base_lcm_1_6B | ~1.6B |

---

## 3.5 Scaling Law 计算 max_steps

```
tokens_per_sentence ≈ 18.5285
sentences_per_step  = batch_size × gradient_accumulation × num_gpus
tokens_per_step     = sentences_per_step × 18.5285
max_steps           = total_tokens / tokens_per_step
```

例：20B tokens，batch_size=32，gradient_accumulation=1，8 GPU：
- sentences_per_step = 256
- tokens_per_step ≈ 4,743
- max_steps ≈ 4,217,000

**Pretrain**：`max_steps = 0.9 × S`，`lr_stage_ratios=[0.1, 0.9, 0.0]`（无 decay）  
**Decay**：`max_steps = S`，`decay_ratio = 0.1`

---

# 附录：sbatch 资源覆盖

| 任务 | 默认 | 1-GPU 覆盖 |
|------|------|------------|
| 预处理（多卡） | 8 GPU, 256G | — |
| 预处理（单卡） | — | `--gres=gpu:1 --mem=64G` |
| Packing | — | `--gres=gpu:1 --mem=128G` |
| LR Sweep | 8 GPU, 256G | 可设 `NPROC=1` 单卡 |
| Pretrain / Decay | 8 GPU, 256G | — |

---

# 附录：完整命令速查表（8 卡）

| 任务 | bash | sbatch |
|------|------|--------|
| **预处理（8 卡）** | salloc 8 卡 → `./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh` | `sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh` |
| **Packing** | salloc 1 卡 → `./quick_runners/packing/pack_parquet.sh` | `sbatch --gres=gpu:1 --mem=128G sbatch_runners/sbatch_bash_runner.sh ./quick_runners/packing/pack_parquet.sh` |
| **LR Sweep** | salloc 8 卡 → `RECIPE=... LR_VALUES=... CUDA_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 ./quick_runners/train/lr_sweep_pretrain.sh` | 同上，用 sbatch 包装 |
| **Pretrain** | salloc 8 卡 → `./quick_runners/train/pretrain.sh` | `sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/pretrain.sh` |
| **Pretrain 恢复** | `RESUME_FROM=... ./quick_runners/train/pretrain_resume.sh` | 同上，sbatch 包装 |
| **Decay** | salloc 8 卡 → `./quick_runners/train/decay.sh` | `sbatch sbatch_runners/sbatch_bash_runner.sh ./quick_runners/train/decay.sh` |
