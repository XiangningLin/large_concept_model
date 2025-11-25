# LCM Training Debug 经验总结

## 问题描述

在 Quest HPC 上使用 submitit 提交 LCM 训练任务时，遇到以下错误：

```
ValueError: The parameters held by `optimizer` must be on a `cuda` device, 
but at least one parameter is on a `cpu` device instead.
```

即使任务已经成功提交到 GPU 节点（qgpu0401），PyTorch 仍然检测不到 CUDA，导致所有模型参数被放置在 CPU 上。

## 调试过程

### 阶段 1: 初步诊断

**现象**：
- 任务成功提交到 SLURM GPU 节点
- 但 `torch.cuda.is_available()` 返回 `False`
- `gang.device` 被设置为 `cpu`
- 所有模型参数在 CPU 上

**初步假设**：
- CUDA 模块未加载
- PyTorch 安装问题
- 环境变量未正确设置

### 阶段 2: 添加详细调试信息

在 `lcm/utils/distributed.py` 的 `init_process_group()` 函数中添加了详细的调试日志：

```python
# 检查点 1: submitit export 之前
- SLURM 环境变量（SLURM_JOB_ID, SLURM_JOB_GPUS, SLURM_GPUS_PER_NODE, SLURM_GRES）
- CUDA 相关环境变量（CUDA_VISIBLE_DEVICES, FAIRSEQ2_DEVICE）
- torch.cuda.is_available() 和 torch.cuda.device_count()
- 如果不可用，检查 torch._C._cuda_getDeviceCount()

# 检查点 2: submitit export 之后
- 环境变量的变化
- LOCAL_RANK（submitit 可能设置）

# 检查点 3: ProcessGroupGang.init_default_process_group 之前
- 最终的环境变量状态

# 检查点 4: ProcessGroupGang.init_default_process_group 之后
- 最终 gang.device 的值
```

### 阶段 3: 关键发现

通过调试日志发现：

1. **SLURM 已分配 GPU**：
   ```
   AllocTRES=...gres/gpu=1  # 从 scontrol show job 确认
   ```

2. **但环境变量未设置**：
   ```
   SLURM_JOB_GPUS: 0
   SLURM_GPUS_PER_NODE: 0
   SLURM_GRES: Not set
   ```

3. **提交脚本的问题**：
   - `#SBATCH --gres=gpu:a100:1` ✅ 存在
   - `#SBATCH --gpus-per-node=0` ⚠️ 冲突
   - `srun` 命令缺少 `--gres` 参数 ❌ **关键问题**

### 阶段 4: 根本原因

**问题根源**：

submitit 生成的提交脚本中：
- `sbatch` 通过 `--gres=gpu:a100:1` 分配了 GPU
- 但 `srun` 命令没有传递 `--gres` 参数
- 导致任务步骤（task step）无法访问 GPU
- PyTorch 检测不到 CUDA 设备

**为什么需要 `srun_args`**：

在 SLURM 中：
- `sbatch` 分配资源给整个作业
- `srun` 在作业内启动任务步骤
- 如果 `srun` 不传递 `--gres`，任务步骤无法访问已分配的 GPU

### 阶段 5: 解决方案

**修改训练脚本**，添加 `srun_args` 参数：

```bash
uv run python -m lcm.train \
    launcher=submitit \
    +post_training=tom_tracking \
    ++launcher.update_parameters.slurm_gres="gpu:a100:1" \
    ++launcher.update_parameters.slurm_srun_args='["--gres=gpu:a100:1"]'  # ← 关键
```

**参考**：`scripts/prepare_tom_tracking.py` 中已经正确使用了这个模式：
```python
update_parameters={
    "slurm_gres": "gpu:a100:1",
    "srun_args": ["--gres=gpu:a100:1"],  # 同时设置两者
}
```

## 解决方案总结

### 1. 添加 `srun_args` 参数

确保 submitit 生成的 `srun` 命令包含 `--gres` 参数：

```bash
++launcher.update_parameters.slurm_srun_args='["--gres=gpu:a100:1"]'
```

### 2. 保留调试信息（可选）

在 `lcm/utils/distributed.py` 中保留调试日志，便于未来排查问题。

### 3. 验证 CUDA 检测

成功后的日志应该显示：
```
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
torch.cuda.get_device_name(0): NVIDIA A100-PCIE-40GB
gang.device: cuda:0
```

## 关键经验教训

### 1. SLURM 资源分配的双重性

- **作业级别**：`sbatch --gres=gpu:a100:1` 分配 GPU 给整个作业
- **任务级别**：`srun --gres=gpu:a100:1` 让任务步骤访问 GPU
- **两者都需要**：只设置 `sbatch` 参数是不够的

### 2. submitit 的参数命名

- `slurm_gres`：用于 `sbatch` 的 `--gres` 参数
- `slurm_srun_args`：用于 `srun` 的额外参数（需要传递 `--gres`）

### 3. 调试策略

- **添加详细日志**：在关键检查点记录环境变量和状态
- **对比参考实现**：查看其他成功运行的脚本（如 `prepare_tom_tracking.py`）
- **验证 SLURM 分配**：使用 `scontrol show job <job_id>` 确认资源分配

### 4. 常见误区

❌ **错误假设**：
- "GPU 节点上一定有 CUDA"
- "只要 `sbatch --gres` 设置了，任务就能用 GPU"
- "`module load cuda` 是必需的"

✅ **正确理解**：
- GPU 节点上，PyTorch 仍需要正确检测 CUDA
- `srun` 必须传递 `--gres` 才能访问 GPU
- 环境变量（如 `SLURM_GRES`）可能不会自动设置

## 相关文件

- **训练脚本**：`quick_runners/post_training/lcm_370m_tom_tracking.sh`
- **配置**：`recipes/train/post_training/tom_tracking.yaml`
- **Launcher 配置**：`recipes/common/launcher/submitit.yaml`
- **设备检测代码**：`lcm/utils/distributed.py`
- **参考实现**：`scripts/prepare_tom_tracking.py`

## 后续优化建议

1. **清理调试日志**：训练稳定后，可以移除详细的调试信息，只保留关键日志
2. **文档化配置**：在 README 中说明 Quest HPC 的特殊配置要求
3. **统一配置模式**：确保所有使用 submitit 的脚本都正确设置 `srun_args`

## 完整执行流程

### 从 Login Node 到 GPU 节点的完整路径

1. **Login Node - 脚本执行**
   ```bash
   bash lcm_370m_tom_tracking.sh
   ```
   - 脚本中的 `#SBATCH` 指令被忽略（因为是 `bash` 运行，不是 `sbatch`）
   - `module load cuda/...` 在 login node 上执行（可能无效）

2. **Login Node - Python 代码执行**
   ```python
   # lcm/train/__main__.py
   launcher = hydra.utils.instantiate(config.launcher)  # 创建 stopes Launcher
   train_module = TrainModule(train_config)
   wait_on = launcher.schedule(train_module)  # 提交 SLURM 任务
   ```

3. **Login Node - submitit 生成提交脚本**
   - submitit 检测到 `cluster="slurm"`
   - 生成 `152354_submission.sh`，包含 `#SBATCH` 指令
   - **关键**：需要同时设置 `slurm_gres` 和 `slurm_srun_args`

4. **Login Node - 提交 SLURM 任务**
   ```bash
   sbatch 152354_submission.sh
   # SLURM 返回：Submitted batch job 152354
   ```

5. **GPU 节点 - SLURM 执行提交脚本**
   - SLURM 在 GPU 节点（qgpu0406）上执行脚本
   - 执行 `srun` 命令（必须包含 `--gres` 参数）

6. **GPU 节点 - submitit 执行训练代码**
   - submitit 反序列化 `TrainModule`
   - 调用 `TrainModule.run()`

7. **GPU 节点 - 初始化 Trainer**
   ```python
   # lcm/train/trainer.py
   self.gang = init_process_group(config, logger)
   ```

8. **GPU 节点 - 设备检测**
   ```python
   # lcm/utils/distributed.py
   # → fairseq2/gang.py
   # → fairseq2/device.py
   device = determine_default_device()  # 检测 CUDA
   ```

### 关键代码路径

```
lcm/train/__main__.py (main)
  → run()
    → launcher.schedule(train_module)
      → submitit 提交 SLURM 任务
        → GPU 节点执行
          → TrainModule.run()
            → get_trainer()
              → LCMTrainerBuilder.__init__()
                → init_process_group()  # ← 设备检测发生在这里
                  → ProcessGroupGang.init_default_process_group()
                    → determine_default_device()
```

## 关于 Overflow 的说明

训练过程中出现的 "Overflow detected" 信息是 **正常的**，不是错误：

- **含义**：Mixed precision training（float16）中的数值溢出检测
- **机制**：`DynamicLossScaler` 自动检测 NaN/Inf 梯度，降低 loss scale
- **处理**：自动忽略溢出步骤的梯度，继续训练
- **结果**：训练正常进行，Loss 持续下降

**示例日志**：
```
Overflow detected at step 1, ignoring gradient, decreasing loss scale from 32768 to 16384.
...
Overflow detected at step 2, ignoring gradient, decreasing loss scale from 0.125 to 0.0625.
Train Metrics (step 50) - Loss: 540.044 | grad_scale: 0.0625
Train Metrics (step 100) - Loss: 317.291 | grad_scale: 0.0625  # Loss 在下降
```

这是 mixed precision training 的标准机制，无需担心。

## 快速检查清单

如果遇到类似问题，按以下步骤检查：

1. ✅ **检查 SLURM 资源分配**
   ```bash
   scontrol show job <job_id> | grep -E "AllocTRES|ReqTRES"
   ```

2. ✅ **检查提交脚本**
   ```bash
   cat executor_logs/.../<job_id>_submission.sh
   # 确认 srun 命令包含 --gres 参数
   ```

3. ✅ **检查环境变量**
   - 查看日志中的 `SLURM_GRES`、`CUDA_VISIBLE_DEVICES`
   - 确认 `torch.cuda.is_available()` 的返回值

4. ✅ **检查 submitit 配置**
   - 确认设置了 `slurm_gres`
   - **关键**：确认设置了 `slurm_srun_args`

5. ✅ **参考成功案例**
   - 对比 `scripts/prepare_tom_tracking.py` 的配置
   - 确认参数命名正确（`slurm_srun_args` 不是 `srun_args`）

