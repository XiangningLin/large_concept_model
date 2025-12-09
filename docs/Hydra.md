# LCM的Hydra配置

## Hydra的调用端：

1. 模型的训练入口：

在 `lcm/train/__main__.py`里面：

```python
@hydra.main(
    version_base="1.2",
    config_path="../../recipes/train",
    config_name="defaults.yaml",
)
def main(config: TrainingConfig) -> None:
    """
    Launch a train module from CLI.

    Example:
    python -m lcm.train +pretrain=mse

    in this example, `pretrain` is a folder under the `recipes` directory and `mse`
    is a yaml file with the trainer configuration.
    This yaml file must be in the `trainer` package (i.e. start with the `# @package trainer`
    hydra directive).
    It must contain a `__trainer__` entry defining the constructor for the trainer.

    You can use `-c job` to see the configuration without running anything. You can use
    `dry_run=true` to initialize the trainer from the configuration and make sure it's correct
    without running the actual training. To debug the jobs, you can use `launcher.cluster=debug`
    """
    asyncio.run(run(config))
```

这里设置的配置路径是`recips/train/`，默认配置为`defaults.yaml`。

2. 模型的评估入口

在代码的`lcm/evaluation/cli/slurm.py`里面：

```python
def main(
    cfg: CliConfig, launcher_opts: LauncherOptions, logger: logging.Logger = logger
) -> None:
    ...launcher = hydra.utils.instantiate(launcher_opts)
```

这似乎不是用Hydra风格的CLI参数来配置的，因为这里不是程序入口。

## Hydra配置树：

### recipes

只有recipe下的配置文件被Hydra加载了（且可以被Hydra加载），在recipe目录下，除了eval没有找到加载端之外，其他文件全部有被Hydra加载的地方。

```
recipes/
├── train/
│   ├── defaults.yaml              ✅ 被加载（作为主配置）
│   ├── pretrain/
│   │   ├── mse_780M.yaml          ✅ 可通过 +pretrain=mse_780M 加载 (base LCM 780M)
│   │   ├── mse.yaml               ✅ 可通过 +pretrain=mse 加载（base LCM 1.6B）
│   │   ├── two_tower_780M.yaml    ✅ 可通过 +pretrain=two_tower_780M 加载
│   │   └── two_tower.yaml          ✅ 可通过 +pretrain=two_tower 加载
│   ├── finetune/
│   │   ├── mse.yaml                ✅ 可通过 +finetune=mse 加载
│   │   └── two_tower.yaml          ✅ 可通过 +finetune=two_tower 加载
│   └── post_training/
│       ├── tom_tracking.yaml       ✅ 可通过 +post_training=tom_tracking 加载
│       └── tom_tracking_4GPU.yaml  ✅ 可通过 +post_training=tom_tracking_4GPU 加载
└── common/
    ├── requirements.yaml           ✅ 通过 defaults 中的 requirements@trainer 加载
    ├── evals.yaml                  ✅ 可能被评估系统使用
    └── launcher/
        ├── standalone.yaml         ✅ 可通过 +launcher=standalone 加载
        └── submitit.yaml            ✅ 可通 +launcher=submitit 加载（默认）
```

## LCM 的 CLI：

### data preprocess：采用fire

fire（[fire repo](https://github.com/google/python-fire)）能够支持在python脚本里运行cli命令。LCM的data preprocess用了fire进行函数参数的cli参数化（在这个场景下，所有函数参数自动成为cli参数，我们可以以覆盖cli参数的方式覆盖函数参数）。

```python
def prepare_fine_web(
    output_dir: str = "output/fine_web",
    num_samples: int = 100,
    start_index: int = 0,
    batch_size: int = 10,
    max_sentence_length: int = int(os.environ.get("MAX_SENTENCE_LENGTH", "256")),
    add_split_column: bool = os.environ.get("ADD_SPLIT_COLUMN", "True").lower() == "true",
    train_ratio: float = float(os.environ.get("TRAIN_RATIO", "0.8")),
    seed: int = int(os.environ.get("SEED", "42")),
):

    ...

if __name__ == "__main__":
    import fire
    fire.Fire(prepare_fine_web)
```

我们使用环境变量来设置这些参数，下面是一个使用prepare_data的例子，包含所有需要改参数的情况（一般来说主要需要修改的地方在于output_dir，如果不输入这些参数将会自动使用默认值）：

```sh
bash prepare_data.sh \
    --num_gpus=4 \
    --num_samples=20000 \
    --output_dir=output/fine_web \
    --batch_size=20 \
    --max_sentence_length=512 \
    --add_split_column=True \
    --train_ratio=0.8 \
    --seed=42
```

在完成prepare data之后，我们还要更新datacard。脚本中自动处理了这个过程。

### trainer：采用Hydra CLI 

LCM采用了先根据所有需要的hydra config的默认值生成顶层配置，再通过+/++等hydra参数来覆盖这些默认值的做法。下面是一个使用LCM风格的Hydra CLI的范例：

```sh
# 最简的例子
python -m lcm.train \
    +pretrain=mse_780M \
    ++trainer.output_dir=checkpoints/my_exp \
    ++trainer.seed=42
```

```sh
# 完整的运行案例
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_GPUS \
    -m lcm.train \
    launcher=standalone \
    +pretrain=mse_780M \
    ++trainer.output_dir=$OUTPUT_DIR \
    ++trainer.experiment_name=$EXPERIMENT_NAME \
    ++trainer.data_loading_config.max_tokens=$MAX_TOKENS \
    ++trainer.use_fsdp=true \
    ++trainer.max_steps=$MAX_STEPS \
    ++trainer.checkpoint_every_n_steps=$CHECKPOINT_EVERY \
    ++trainer.save_model_every_n_steps=$CHECKPOINT_EVERY \
    ++trainer.publish_metrics_every_n_steps=100
```

代码运行时会先通过`hydra.main()`进入recipes/train/defaults.yaml，加载所有默认配置组，看到有+pretrain，说明我们选择了recipes/train下的pretrain/mse_780M选项。然后由于每个train下面的yaml文件都会有`@package trainer`作为修饰，因此我们可以把`trainer.output_dir`理解为`pretrain.output_dir`，从而进行进一步的覆盖。

LCM用了一个schema TrainingConfig来包装和validate我们的Hydra Config。理论上我们只需要修改recipes/train下的文件中存在的那些参数就可以控制训练流程（前提是datacard和数据文件已经正确配置和存在）。

在pretrain.sh里面，我们也通过环境变量来实现hydra参数的覆盖，下面是一个完整参数的例子：

```sh
# 完整参数示例
bash pretrain.sh \
    --num_gpus=4 \
    --data_dir=output/fine_web_data \
    --output_dir=checkpoints/my_experiment \
    --experiment_name=my_experiment_name \
    --max_tokens=6000 \
    --max_steps=100000 \
    --data_name=fine_web_edu

# 最小参数示例（使用所有默认值）
bash pretrain.sh

# 只指定必要参数
bash pretrain.sh \
    --num_gpus=2 \
    --data_name=fine_web_edu

# 常用配置示例
bash pretrain.sh \
    --num_gpus=4 \
    --output_dir=checkpoints/fine_web_780m \
    --experiment_name=fine_web_780m_exp \
    --max_steps=50000 \
    --data_name=fine_web_edu
```

### eval：混合Hydra CLI与argparse

LCM的eval功能里，Hydra只用于在slurm（submitit）模式下来读取slurm有关的参数。如果是在已有的计算节点上运行任务的话，可以忽略这个设置，把它当作纯粹依赖argparse的部分。


