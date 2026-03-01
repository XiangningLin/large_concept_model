# LCM 项目多进程/多卡支持分析报告

> 日期：2026-02-25  
> 范围：数据预处理 → Packing → 训练 → LR Sweep → 评估  
> 环境：fairseq2 + stopes + PyTorch Distributed（无 accelerate）

---

## 总览

| 功能模块 | 多卡支持 | 实现方式 | 当前状态 |
|----------|---------|---------|---------|
| **数据预处理 (FineWeb)** | 否 | 手动分块 + 外部并行 | 单进程，需手动分 chunk |
| **数据预处理 (C4/Wiki/ToM)** | 是 | stopes PartitionedDataMapper | 自动多 GPU/SLURM 分片 |
| **Packing** | 否 | 单进程 CPU | GPU 几乎不用 |
| **训练** | **是** | torchrun + fairseq2 FSDP/DDP | 完整支持，生产级别 |
| **LR Sweep** | 部分 | torchrun (默认单卡) | 支持但默认关闭 |
| **评估 (本地)** | **否** | 强制 WORLD_SIZE=1 | 硬编码单卡 |
| **评估 (SLURM)** | 部分 | stopes 数据分片 | 支持但限制每 shard 1 GPU |
| **合并脚本** | 否 | Shell 串行 | 纯 I/O，不需要多卡 |

---

## 1. 数据预处理

### 1.1 `prepare_fine_web.py` — 单进程，无分布式

**结论：完全不支持多进程/多卡并行。**

核心证据：

- **GPU 选择**：`torch.device("cuda")` 硬编码使用 GPU 0，无 `local_rank` 机制
  ```python
  # scripts/prepare_fine_web.py L313-318
  if torch.cuda.is_available():
      device = torch.device("cuda")   # 永远 cuda:0
  ```

- **无分布式初始化**：没有 `torch.distributed.init_process_group()`、`accelerate`、`torchrun` 相关代码
- **无 rank/world_size 概念**：没有环境变量读取 (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`)
- **SONAR 模型加载**：单 GPU 加载，无 DDP/FSDP 封装
- **检查点系统**：写入固定路径 `output_dir/checkpoints/`，多进程会冲突

**当前"多卡"方案**：手动启动多个独立进程（不是多进程协作）

```bash
# 实际是 N 个完全独立的 Python 进程，各用不同 GPU
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_fine_web.py --start_index=0      --num_samples=920000 --output_dir=chunk_0
CUDA_VISIBLE_DEVICES=1 python scripts/prepare_fine_web.py --start_index=920000  --num_samples=920000 --output_dir=chunk_1
# ...
```

这不是"多进程运行"，而是"多次单进程运行"。进程之间无通信、无协调、无共享状态。

**与 SentenceSSM 的对比**：SentenceSSM 的 `data_processing_pipeline_token_parquet_acc.py` 使用 `accelerate` 框架，支持真正的多 rank 并行：
- 通过 `accelerator.process_index` 获取 rank
- 输出 `rank=0/part-*.parquet`, `rank=1/part-*.parquet`
- 所有 rank 自动分配 GPU
- 需要后续 merge 步骤

### 1.2 `prepare_c4_parallel.py` / `prepare_wikipedia.py` / `prepare_tom_tracking.py` — 真正的多卡

这三个脚本使用 **stopes 框架**，支持完整的多卡并行：

```python
# scripts/prepare_c4_parallel.py
from stopes.core.launcher import Launcher
from stopes.modules.partitioned_data_mapper import stopes_data_mapper

# 自动分 num_shards 个任务，每个任务 1 GPU
req = Requirements(gpus_per_node=1, ...)
stopes_wrapped = stopes_data_mapper(req, {...})(FullPipeline)
asyncio.run(launcher.schedule(stopes_module))
```

特性：
- `num_shards` 控制并行度（如 64 个 shard 分配到 4 个 GPU 轮流执行）
- 支持本地多 GPU（`use_slurm=False`）和 SLURM 集群（`use_slurm=True`）
- stopes 自动处理任务调度、容错、输出合并

**为什么 `prepare_fine_web.py` 没有用 stopes？**  
因为 `prepare_fine_web.py` 是从简单脚本演化而来，加了断点恢复和流式处理等功能，但没有重构为 stopes 架构。`prepare_c4_parallel.py` 才是推荐的多卡预处理模板。

---

## 2. Packing (`pack_parquet.py`)

**结论：不需要多卡，CPU 足够。**

- GPU 使用极少：仅在启动时调用一次 SONAR 编码 "End of text." 字符串（~5 秒）
- 主要工作是 CPU/IO 密集：读 Parquet、贪心打包、写 Parquet
- 当前为单进程串行处理
- 如需加速，应考虑 `multiprocessing`（CPU 多进程），而非多 GPU

---

## 3. 训练 Pipeline

**结论：完整支持多卡，是整个项目中分布式支持最成熟的部分。**

### 3.1 启动方式

两种模式，均使用 `torchrun`（而非 `accelerate`）：

| 模式 | 命令 | 适用场景 |
|------|------|---------|
| **standalone** | `torchrun --standalone --nnodes=1 --nproc-per-node=N -m lcm.train launcher=standalone` | 单节点多卡 |
| **submitit** | `python -m lcm.train launcher=submitit` | SLURM 集群多节点 |

### 3.2 分布式训练机制

| 组件 | 实现 |
|------|------|
| **进程组** | fairseq2 `ProcessGroupGang`（封装 `torch.distributed`） |
| **模型并行** | FSDP（默认）或 DDP；由 `trainer.use_fsdp` 控制 |
| **FSDP 粒度** | `"model"` / `"stack"` / `"layer"` 三种 wrap 粒度可选 |
| **梯度同步** | 自动（FSDP/DDP）+ `no_sync()` 优化梯度累积 |
| **梯度一致性检查** | `check_gradient_norms()` 验证所有 rank 梯度范数一致 |
| **Checkpoint** | FSDP: 每 rank 各存自己的 shard；DDP: rank 0 存全量 |
| **Token 统计** | `gang.all_reduce(total_targets, SUM)` 跨 rank 累加 |
| **日志/WandB** | 仅 rank 0 输出 |

### 3.3 数据加载的多 rank 分片

数据分片有两种策略，自动选择：

**策略 A — Fragment 级分片**（默认，大数据集）：

```python
# lcm/datasets/dataloading.py L228-233
if not sharding_in_memory:
    pipeline = pipeline.shard(
        shard_idx=rank,
        num_shards=world_size,
        allow_uneven=not self.loading_config.even_sharding,
    )
```

每个 rank 只加载自己负责的 parquet fragment，IO 压力分散。

**策略 B — In-memory 分片**（小数据集或 `even_sharding=True`）：

```python
# L297-302
if sharding_in_memory:
    pipeline = pipeline.shard(shard_idx=rank, ...)
```

所有 rank 加载全部 fragment，在 batch 级别分片。保证每个 rank 看到完全相同数量的 batch。

**Seed 处理**：
- Fragment 级分片：`seed + rank * 100_000`（每 rank 不同 shuffle）
- In-memory 分片：所有 rank 用同一 seed（保证分片前顺序一致）

### 3.4 验证（Validation）

验证循环支持多卡，且处理了数据不均匀问题：

```python
# lcm/train/trainer.py L1095-1112
try:
    batch = next(data_iter)
    true_batch = 1
except StopIteration:
    batch = next(data_dummy_iter)  # 用 dummy batch 保持 rank 同步
    true_batch = 0
total_nb_batches = all_sum(self.gang, true_batch)
if bool(total_nb_batches == 0):
    break
```

当某个 rank 数据耗尽时，用 dummy batch 代替，防止集体通信 hang。

---

## 4. LR Sweep

**结论：支持多卡但默认关闭。**

`quick_runners/train/lr_sweep_pretrain.sh` 通过环境变量控制：

```bash
# 默认值
NPROC=${NPROC:-1}         # 默认单卡
CUDA_DEVICES=${CUDA_DEVICES:-0}
```

调用方式：
```bash
# 单卡（默认）
./quick_runners/train/lr_sweep_pretrain.sh

# 多卡
NPROC=4 CUDA_DEVICES=0,1,2,3 ./quick_runners/train/lr_sweep_pretrain.sh
```

**注意**：脚本强制 `++trainer.use_fsdp=false`（L68），多卡时使用 DDP。对于大模型（780M+）可能 OOM，需要手动移除该覆盖。

**LR 值之间是串行的**（不是并行），即一次只跑一个 LR。

---

## 5. 评估（Evaluation）

### 5.1 本地评估

**结论：硬编码单卡，不支持多 GPU。**

关键代码：

```python
# lcm/evaluation/utils/common.py L514-523
def set_torch_variables() -> None:
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
```

无论有多少 GPU，本地评估始终以 `WORLD_SIZE=1` 运行。

### 5.2 SLURM 评估

支持数据分片，但有限制：

```python
# lcm/evaluation/arun.py L50-53
assert self.nshards == 1 or self.requirements.gpus_per_node == 1
# 要么：1 shard + 多 GPU（但实际只用 1 GPU）
# 要么：多 shard + 每 shard 1 GPU
```

不支持单 shard 跨多 GPU 的模型并行评估。

---

## 6. 合并脚本 (`merge_packed_chunks.sh`)

**结论：纯 Shell 串行 cp/mv，无需多进程。**

两种模式：
- `--mode chunk`：合并 packed 后的 Hive 分区 chunk
- `--mode preprocess`：合并 flat `data.parquet` chunk

---

## 7. 差距分析与建议

### 7.1 最大差距：`prepare_fine_web.py` 缺乏真正的多卡支持

这是整个 pipeline 中最耗时的步骤（9.2M 样本，单卡可能需要数天），但没有原生多卡支持。

**三种改进路径**（工作量递增）：

| 方案 | 改动量 | 描述 |
|------|--------|------|
| **A. 保持现状** | 无 | 继续用 `--start_index` 手动分块 + `merge_packed_chunks.sh --mode preprocess` 合并。可工作，但需人工协调。 |
| **B. 添加 `--rank`/`--world_size` 参数** | 小 | 脚本内自动计算 `start_index = rank * (total // world_size)`，自动设置 `output_dir/chunk_{rank}/`。支持 `torchrun` 启动。 |
| **C. 迁移到 stopes 框架** | 中 | 参照 `prepare_c4_parallel.py`，用 `stopes_data_mapper` + `HFInputConfig` 重写。获得自动调度、容错、SLURM 支持。 |

**推荐方案 B**：改动最小，效果显著。只需增加 ~20 行代码即可让脚本支持 `torchrun --nproc-per-node=4` 直接启动。

### 7.2 评估模块的单卡限制

`set_torch_variables()` 硬编码 `WORLD_SIZE=1` 是一个设计缺陷。对于大模型（如 1.6B），评估时可能需要多卡加载。建议：
- 检测 `torchrun` 环境变量是否已设置，如果已设置则不覆盖
- 或提供 `--distributed` 命令行参数

### 7.3 LR Sweep 的 FSDP 限制

`use_fsdp=false` 限制了大模型的多卡 sweep。建议根据模型大小自动选择：
- 小模型（≤370M）：DDP（默认关闭 FSDP）
- 大模型（>370M）：保留 FSDP

---

## 8. 架构对比：LCM vs SentenceSSM

| 方面 | LCM | SentenceSSM |
|------|-----|-------------|
| **预处理多卡** | 手动分块（FineWeb）/ stopes（C4/Wiki） | accelerate 原生多 rank |
| **预处理输出** | `chunk_N/data.parquet` | `rank=N/part-*.parquet` |
| **需要 merge** | 分块后需要 merge（preprocess 模式） | 需要 merge（rank 模式） |
| **训练框架** | fairseq2 (FSDP/DDP) | HuggingFace accelerate |
| **训练启动** | `torchrun` / `submitit` | `accelerate launch` |
| **评估多卡** | 不支持（本地） | N/A |

---

## 9. 结论

**训练（包括验证）的多卡支持是完善的**，使用 fairseq2 + torchrun 的标准方案，FSDP/DDP 均可用。

**预处理是主要短板**：`prepare_fine_web.py` 完全不支持多进程，只能通过外部手动分块模拟并行。项目中已有更好的多卡预处理模板（`prepare_c4_parallel.py`），但 FineWeb 脚本没有采用。

**评估的多卡支持存在人为限制**：分布式基础设施已经就绪（`all_reduce`, `gather_object` 等），但启动入口硬编码了单卡。
