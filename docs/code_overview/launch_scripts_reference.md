# LCM 项目启动脚本与运行入口参考

本文档归纳 large_concept_model 项目中可供用户运行的启动脚本、对应命令、Hydra 配置组合，以及各自支持的动作。

---

## 一、核心 Python 入口（主启动脚本）

### 1. 训练：`lcm.train`

**命令形式：**
```bash
python -m lcm.train [launcher=...] [+recipe_group=recipe_name] [++override.key=value ...]
```

**入口文件：** `lcm/train/__main__.py`  
**配置路径：** `recipes/train/`（Hydra `config_path`）  
**默认配置：** `recipes/train/defaults.yaml`

#### Hydra 配置结构

| 配置项 | 说明 |
|--------|------|
| `launcher` | 启动器：`submitit`（SLURM）、`standalone`（本地 torchrun） |
| `+pretrain=mse` | 预训练 MSE LCM（1.6B） |
| `+pretrain=two_tower` | 预训练 Two-tower diffusion LCM（1.6B） |
| `+pretrain=mse_780M` | 预训练 MSE LCM（780M） |
| `+pretrain=two_tower_780M` | 预训练 Two-tower diffusion LCM（780M） |
| `+finetune=mse` | 微调 MSE LCM |
| `+finetune=two_tower` | 微调 Two-tower diffusion LCM |
| `+post_training=tom_tracking` | Tom Tracking 单卡训练（370M） |
| `+post_training=tom_tracking_4GPU` | Tom Tracking 4 卡训练（370M） |
| `++trainer.output_dir` | 输出目录（必填） |
| `++trainer.experiment_name` | 实验名称 |
| `++trainer.model_config_or_name` | 微调时指定预训练模型（fairseq2 asset 名） |
| `+trainer.use_submitit=false` | 使用 standalone 时关闭 submitit |

#### 支持的动作

| 动作 | 命令示例 | 说明 |
|------|----------|------|
| **预训练 MSE LCM (1.6B)** | `python -m lcm.train +pretrain=mse ++trainer.output_dir="checkpoints/mse_lcm" ++trainer.experiment_name=training_mse_lcm` | 4 节点 × 8 GPU，SLURM 提交 |
| **预训练 MSE LCM (本地)** | `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 -m lcm.train launcher=standalone +pretrain=mse ++trainer.output_dir="checkpoints/mse_lcm" +trainer.use_submitit=false` | 本地 2 GPU |
| **预训练 Two-tower (1.6B)** | `python -m lcm.train +pretrain=two_tower ++trainer.output_dir="checkpoints/two_tower_lcm" ++trainer.experiment_name=training_two_tower_lcm` | SLURM |
| **微调 Two-tower** | `python -m lcm.train +finetune=two_tower ++trainer.output_dir="checkpoints/finetune_two_tower" ++trainer.model_config_or_name=my_pretrained_two_tower` | 需先注册预训练 checkpoint |
| **微调 MSE** | `python -m lcm.train +finetune=mse ++trainer.output_dir="checkpoints/finetune_mse" ++trainer.model_config_or_name=my_pretrained_mse_lcm` | 同上 |
| **Tom Tracking (4 GPU)** | `torchrun --standalone --nnodes=1 --nproc-per-node=4 -m lcm.train launcher=standalone +post_training=tom_tracking_4GPU ++trainer.output_dir="checkpoints/tom_tracking" +trainer.use_submitit=false` | 370M 架构，需 tom_tracking_data datacard |

#### 常用覆盖示例

```bash
# 覆盖 SLURM 资源
++trainer.requirements.timeout_min=100
++trainer.requirements.cpus_per_task=8
++launcher.partition=$partition_name

# 覆盖数据与训练
++trainer.data_loading_config.max_tokens=1000
++trainer.max_steps=500
++trainer.validate_every_n_steps=100
```

---

### 2. 评估：`lcm.evaluation`

**命令形式：**
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.evaluation \
  --predictor <predictor_name> \
  --model_card <path> \
  --tasks <task_name> \
  --task_args '<json>' \
  --dump_dir <output_dir> \
  [--launcher standalone|submitit] \
  [其他参数...]
```

**入口文件：** `lcm/evaluation/__main__.py`  
**配置方式：** CLI 参数（非 Hydra）

#### Predictor 与模型对应

| Predictor | 模型类型 | 主要参数 |
|-----------|----------|----------|
| `base_lcm` | Base MSE LCM | `--model_card` |
| `two_tower_diffusion_lcm` | Two-tower diffusion LCM | `--model_card` |
| `llama3` | LLaMA 3.x | `--model_name` |
| `huggingface` | HF AutoModel | `--model_name` |
| `gemma` | Gemma | `--model_name` |

#### 支持的动作

| 动作 | 命令示例 |
|------|----------|
| **评估 Two-tower LCM** | `torchrun ... -m lcm.evaluation --predictor two_tower_diffusion_lcm --model_card ./checkpoints/.../model_card.yaml --tasks finetuning_data_lcm.validation --task_args '{"max_gen_len": 10}' --dump_dir evaluation_outputs/two_tower --inference_timesteps 40 --guidance_scale 3` |
| **评估 Base MSE LCM** | `torchrun ... -m lcm.evaluation --predictor base_lcm --sample_latent_variable False --model_card ./checkpoints/.../model_card.yaml --tasks finetuning_data_lcm.validation --dump_dir evaluation_outputs/mse` |
| **评估 LLM (Llama)** | `torchrun ... -m lcm.evaluation --predictor llama3 --model_name meta-llama/Llama-3.1-8B-Instruct --tasks cnn_dailymail_llm.test --task_args '{"max_gen_len": 200}' --dataset_dir jsonl_dataset/cnn_dailymail --dump_dir output_results` |

#### 常用任务名

| 任务 | 说明 |
|------|------|
| `finetuning_data_lcm.validation` | 微调数据验证集 |
| `lcm_generation` | 通用 LCM 生成任务 |
| `cnn_dailymail_llm.test` | CNN/DM 摘要（LLM） |
| `xsum_llm.test` | XSum 摘要（LLM） |
| `xlsum_llm.{lang}.{split}` | XLSum 多语言摘要 |

---

## 二、数据准备脚本

### 3. `scripts/prepare_wikipedia.py`（当前为 Tom Tracking 预处理）

**命令形式：**
```bash
uv run --extra data scripts/prepare_wikipedia.py <output_dir> [--split=...] [--num_shards=...] [--batch_size=...] [--cache_dir=...] [--use_natural_language=True] [--add_split_column=True] [--train_ratio=0.8]
```

**说明：** 当前实现针对 PrincetonPLI/LongProc 的 Tom Tracking 数据：下载、分句、SONAR 嵌入，输出 Parquet。

**典型用法：**
```bash
uv run --extra data scripts/prepare_wikipedia.py /output/dir/for/the/data
# 或带参数
scripts/prepare_wikipedia.py --output_dir=/path/to/output --split=tom_tracking_0.5k --num_shards=1 --batch_size=10 --add_split_column=True --train_ratio=0.8
```

---

### 4. `scripts/fit_embedding_normalizer.py`

**命令形式：**
```bash
python scripts/fit_embedding_normalizer.py --ds dataset1:4 dataset2:1 dataset3:10 --save_path "path/to/new/normalizer.pt" [--max_nb_samples 1000000]
```

**说明：** 在指定数据集混合上拟合 embedding 归一化器，供 `lcm/cards/sonar_normalizer.yaml` 等使用。

---

### 5. `scripts/prepare_tom_tracking.py`

**命令形式：**
```bash
python scripts/prepare_tom_tracking.py \
  --output_dir="$DATA_OUTPUT_DIR" \
  --split=tom_tracking_0.5k \
  --num_shards=1 \
  --batch_size=10 \
  --cache_dir="$CACHE_DIR" \
  --use_natural_language=True \
  --add_split_column=True \
  --train_ratio=0.8
```

**说明：** 与 `prepare_wikipedia.py` 类似，专门用于 Tom Tracking 数据准备（LongProc）。

---

### 6. `examples/evaluation/prepare_evaluation_data.py`

**命令形式：**（在 `examples/evaluation/` 目录下运行，或 `cd examples/evaluation && uv run ...`）
```bash
# Step 1: 下载并解析为 JSONL
uv run --extra data prepare_evaluation_data.py prepare_data \
  --dataset_name=cnn_dailymail \
  --output_dir=jsonl_dataset \
  --source_text_column=article \
  --target_text_column=highlights \
  --version=3.0.0 \
  [--prompt_prefix="..."] [--prompt_suffix="..."]

# Step 2: 分句 + SONAR 嵌入
uv run --extra data prepare_evaluation_data.py embed \
  --input_path=jsonl_dataset/cnn_dailymail/test.jsonl \
  --source_text_column=prompt \
  --target_text_column=answer \
  --output_dir=parquet_dataset/cnn_dailymail \
  --lang=eng_Latn \
  --mode=local
```

**说明：** 为评估准备数据：下载 HF 数据集 → JSONL → 分句 + 嵌入 → Parquet。

---

## 三、Shell 封装脚本

| 脚本 | 作用 | 典型用法 |
|------|------|----------|
| `prepare_data_and_pretrain.sh` | 准备 Tom Tracking 数据 + 2 GPU 训练 780M | `bash prepare_data_and_pretrain.sh` |
| `quick_runners/post_training/lcm_370m_tom_tracking_4GPU_standalone.sh` | 4 GPU standalone 训练 Tom Tracking 370M | `bash quick_runners/post_training/lcm_370m_tom_tracking_4GPU_standalone.sh` |
| `quick_runners/post_training/lcm_370m_tom_tracking_4GPU.sh` | SLURM sbatch 提交 4 GPU Tom Tracking | `sbatch quick_runners/post_training/lcm_370m_tom_tracking_4GPU.sh` |
| `run_prepare_wikipedia.sh` | 封装 `prepare_wikipedia.py` 调用 | 需根据实际脚本调整 |
| `check_env.sh` | 检查环境（Python、fairseq2、libsndfile 等） | `bash check_env.sh` |
| `reinstall_venv.sh` | 重建虚拟环境 | `bash reinstall_venv.sh` |

---

## 四、训练 Recipe 与 Hydra 组合速查

| Recipe 组 | Recipe 名 | 模型 | 用途 |
|-----------|-----------|------|------|
| `+pretrain=` | `mse` | base_lcm_1_6B | MSE 预训练 1.6B |
| `+pretrain=` | `two_tower` | two_tower_diffusion_lcm_1_6B | Two-tower 预训练 1.6B |
| `+pretrain=` | `mse_780M` | base_lcm_780M | MSE 预训练 780M |
| `+pretrain=` | `two_tower_780M` | two_tower_diffusion_lcm_780M | Two-tower 预训练 780M |
| `+finetune=` | `mse` | 从 model_config_or_name 加载 | MSE 微调 |
| `+finetune=` | `two_tower` | 从 model_config_or_name 加载 | Two-tower 微调 |
| `+post_training=` | `tom_tracking` | base_lcm_370M | Tom Tracking 单卡 |
| `+post_training=` | `tom_tracking_4GPU` | base_lcm_370M | Tom Tracking 4 卡 |

---

## 五、Datacard 依赖

训练与评估依赖 datacard 配置，数据路径在 `lcm/datacards/` 中定义。常用 datacard 名：

- `pretraining_data`：预训练数据
- `finetuning_data`：微调数据
- `tom_tracking_data`：Tom Tracking 数据

需在 datacard 中配置 `parquet_path` 等，指向实际数据路径。

---

## 六、评估前准备

评估 ROUGE 前需下载 NLTK 数据：
```bash
python -m nltk.downloader punkt_tab
```

---

## 七、完整流程示例

### 预训练 → 微调 → 评估

```bash
# 1. 准备数据（示例：Tom Tracking）
env -u VIRTUAL_ENV .venv/bin/python scripts/prepare_tom_tracking.py \
  --output_dir=/path/to/data --split=tom_tracking_0.5k \
  --add_split_column=True --train_ratio=0.8

# 2. 更新 lcm/datacards/ 中的路径

# 3. 预训练
python -m lcm.train +pretrain=mse \
  ++trainer.output_dir="checkpoints/mse_lcm" \
  ++trainer.experiment_name=training_mse_lcm

# 4. 注册 checkpoint 到 lcm/cards/mycards.yaml

# 5. 微调
python -m lcm.train +finetune=mse \
  ++trainer.output_dir="checkpoints/finetune_mse" \
  ++trainer.model_config_or_name=my_pretrained_mse_lcm

# 6. 评估
torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.evaluation \
  --predictor base_lcm --model_card checkpoints/finetune_mse/checkpoints/step_1000/model_card.yaml \
  --tasks finetuning_data_lcm.validation --dump_dir evaluation_outputs/mse
```
