# LCM 项目代码概览

本目录包含 large_concept_model 项目的代码结构与运行入口说明。

## 文档索引

| 文档 | 说明 |
|------|------|
| [launch_scripts_reference.md](./launch_scripts_reference.md) | **启动脚本与运行入口参考**：所有可执行入口、命令、Hydra 配置、支持的动作 |
| [hydra_difference_vs_sentence_ssm.md](./hydra_difference_vs_sentence_ssm.md) | **Hydra 配置差异对比**：LCM 与 SentenceSSM 的 `+`/`++` 与 `config=...` 用法差异及代码原因 |
| [cli_params_reference.md](./cli_params_reference.md) | **CLI 参数原子合集**：预处理、预训练、Finetune、评估任务的完整 CLI 参数表及脚本坐标 |

## 快速入口

- **训练**：`python -m lcm.train +pretrain=mse ++trainer.output_dir=...`
- **评估**：`torchrun ... -m lcm.evaluation --predictor base_lcm --model_card ... --tasks ... --dump_dir ...`
- **数据准备**：`scripts/prepare_wikipedia.py`、`scripts/prepare_tom_tracking.py`、`scripts/fit_embedding_normalizer.py`
- **评估数据准备**：`examples/evaluation/prepare_evaluation_data.py prepare_data` / `embed`

详见 [launch_scripts_reference.md](./launch_scripts_reference.md)。

