# Logic Chains（逻辑链文档）

本目录存放 LCM 各模块的**逻辑链分析**文档，用于梳理从入口到输出的数据流、分支与依赖关系。

## 已整理

### 数据预处理

| 文档 | 说明 |
|------|------|
| [fineweb_preprocessing.md](./fineweb_preprocessing.md) | FineWeb 数据预处理：`prepare_data.sh` → `prepare_fine_web.py` → Parquet + `update_datacards.py` 的完整逻辑链 |

### 训练能力与逻辑链

| 文档 | 说明 |
|------|------|
| [training_summary.md](./training_summary.md) | **训练能力总结**：Base LCM 与 Two-Tower 支持的学习率调度（cosine / WSD 等）、断点恢复、预训练/微调、W&B 记录等 |
| [train_base_lcm.md](./train_base_lcm.md) | **Base LCM 训练逻辑链**：入口 → Builder → 单步训练 → Criterion（next_sentence_mse / target_mse）→ 数据流与 checkpoint |
| [train_two_tower_lcm.md](./train_two_tower_lcm.md) | **Two-Tower Diffusion LCM 训练逻辑链**：入口 → Builder → Criterion（加噪/去噪、step 采样）→ 模型 forward → 数据流与 checkpoint |

### 对比分析

| 文档 | 说明 |
|------|------|
| [lcm_vs_our_code_comparison.md](./lcm_vs_our_code_comparison.md) | **LCM 与我们的代码（SentenceSSM）能力对比**：预处理（智能引号、start_index、UNK 过滤等）与训练（token milestones、decay_from_pretrain、decay_ratio 等）的差异表，以及 LCM 目前做不到的能力汇总 |

## 计划可补充

- C4 / Wikipedia / ToM 等数据预处理的逻辑链
- 评估入口 `lcm.evaluation` 的逻辑链
