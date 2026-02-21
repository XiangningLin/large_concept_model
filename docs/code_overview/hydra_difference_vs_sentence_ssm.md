# LCM 与 SentenceSSM 的 Hydra 配置用法对比

本文档说明 LCM 与 SentenceSSM 在 Hydra 配置系统上的差异，以及导致 `+`/`++` 与 `config=...` 不同用法的代码原因。

---

## 一、用法对比速览

| 场景 | LCM | SentenceSSM |
|------|-----|-------------|
| 选择训练类型/任务 | `+pretrain=mse` | `task=train` |
| 选择模型 | `+pretrain=mse_780M`（recipe 内含模型） | `model=mamba2_370m` |
| 覆盖嵌套参数 | `++trainer.output_dir=...` | `training_mode.learning_rate=1e-4` |
| 覆盖顶层参数 | `++trainer.max_steps=500` | `training_mode.max_steps=500` |

---

## 二、根本原因：`defaults` 结构不同

### 2.1 LCM 的 defaults 结构

**文件：** `recipes/train/defaults.yaml`

```yaml
defaults:
  - launcher: submitit
  - requirements@trainer
  - _self_
dry_run: false
trainer:
  output_dir: ???   # 必填，无默认
launcher:
  log_folder: ${trainer.output_dir}/executor_logs
  config_dump_dir: ${trainer.output_dir}/config_logs
```

**要点：**

- `defaults` 中**没有** `pretrain`、`finetune`、`post_training` 等 recipe 组。
- 用户必须通过 `+pretrain=mse` **显式添加** 一个 recipe，否则 `trainer` 下没有模型、优化器、数据等配置。
- `+` 表示 **add**：向 defaults 中**追加**一个 config group，而不是替换已有项。
- `++trainer.output_dir=...` 中的 `++` 表示 **override**：强制覆盖已有值（包括 `???` 占位符）。

### 2.2 SentenceSSM 的 defaults 结构

**文件：** `src/configs/config.yaml` 或 `baselines/token_models/configs/config.yaml`

```yaml
defaults:
  - _self_
  - task: preprocess
  - data: fine_web
  - model: mamba2_780m
  - training_mode: pretrain
```

**要点：**

- `defaults` 中**已经包含** `task`、`data`、`model`、`training_mode` 四个 config group。
- 用户通过 `task=train`、`model=mamba2_370m` 等 **替换** 默认选择，无需 `+`。
- `task=train` 等价于：把 defaults 里的 `task: preprocess` 换成 `task: train`。
- 覆盖嵌套参数用 `training_mode.learning_rate=1e-4` 即可，一般不需要 `++`。

---

## 三、Hydra 语法简要说明

| 语法 | 含义 | 典型用途 |
|------|------|----------|
| `key=value` | 覆盖已有配置项 | 替换 defaults 中已存在的值 |
| `+key=value` | 添加新的 config group 或新键 | 添加 defaults 中**没有**的 config group |
| `++key=value` | 强制覆盖（删除再设置） | 覆盖 `???`、覆盖只读/结构化配置 |

---

## 四、代码层面的差异

### 4.1 LCM：recipe 与 trainer 的合并方式

**入口：** `lcm/train/__main__.py`

```python
@hydra.main(
    version_base="1.2",
    config_path="../../recipes/train",
    config_name="defaults.yaml",
)
def main(config: TrainingConfig) -> None:
    ...
```

**Recipe 文件：** `recipes/train/pretrain/mse.yaml`

```yaml
# @package trainer
_trainer_: lcm.train.lcm.trainer.prepare_lcm_trainer
output_dir: ??
model_arch: base_lcm_1_6B
lr: 0.0004
...
```

- `@package trainer`：该 yaml 的内容会**合并到**顶层 `trainer` 节点下。
- `+pretrain=mse` 会加载 `recipes/train/pretrain/mse.yaml`，并将其内容写入 `config.trainer`。
- 因此覆盖时要用 `++trainer.output_dir`、`++trainer.max_steps` 等，路径以 `trainer` 为根。

### 4.2 SentenceSSM：扁平化的 config group 选择

**入口：** `baselines/token_models/runners/run_training.py`

```python
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    ...
```

**Config 结构：** 各 group 平铺在顶层：

- `config.task` ← `task/train.yaml` 或 `task/preprocess.yaml`
- `config.data` ← `data/fine_web.yaml` 等
- `config.model` ← `model/mamba2_780m.yaml` 等
- `config.training_mode` ← `training_mode/pretrain.yaml` 等

- `task=train`：把 `task` 从 `preprocess` 换成 `train`。
- `training_mode.learning_rate=1e-4`：直接覆盖 `training_mode` 下的 `learning_rate`。

---

## 五、为何 LCM 必须用 `+` 和 `++`

1. **`+` 用于添加 recipe**
   - `defaults` 中没有 `pretrain`，必须用 `+pretrain=mse` 添加。
   - 若写成 `pretrain=mse`，Hydra 会尝试覆盖一个不存在的顶层键，行为可能不符合预期。

2. **`++` 用于覆盖必填与结构化配置**
   - `trainer.output_dir: ???` 是必填占位符，普通 `=` 有时无法正确覆盖。
   - LCM 使用结构化 schema（`TrainingConfig`），`++` 可确保覆盖生效。

---

## 六、为何 SentenceSSM 只需 `key=value`

1. **所有 config group 已在 defaults 中**
   - `task`、`data`、`model`、`training_mode` 都有默认项。
   - 切换时用 `task=train`、`model=mamba2_370m` 即可，无需 `+`。

2. **无 `???` 占位符**
   - 没有强制必填的顶层占位符，普通 `key=value` 或 `group.key=value` 即可覆盖。

3. **结构更扁平**
   - 覆盖路径如 `training_mode.learning_rate`，不依赖 `@package` 合并，逻辑更直观。

---

## 七、迁移时的对应关系

若从 SentenceSSM 风格迁移到 LCM，可参考：

| SentenceSSM | LCM 近似写法 |
|-------------|--------------|
| `task=train` | 已隐含在 `+pretrain` 中（训练任务） |
| `model=mamba2_370m` | `+pretrain=mse_780M` 或 `+pretrain=two_tower_780M`（recipe 内含模型） |
| `training_mode.learning_rate=1e-4` | `++trainer.lr=0.0001` |
| `training_mode.max_steps=500` | `++trainer.max_steps=500` |
| `data.target_path=/path` | 需在 datacard 中配置，或通过 `++trainer.data_loading_config...` 等覆盖 |

---

## 八、总结

| 维度 | LCM | SentenceSSM |
|------|-----|-------------|
| **defaults 设计** | 只含 launcher、requirements，不含 recipe | 含 task、data、model、training_mode |
| **选择任务/模型** | `+pretrain=mse`（添加 recipe） | `task=train`、`model=mamba2_370m`（替换 group） |
| **覆盖参数** | `++trainer.xxx=value` | `training_mode.xxx=value` 或 `group.key=value` |
| **原因** | recipe 为可选添加；`output_dir` 等为必填 | 所有 group 已在 defaults 中；结构更扁平 |
