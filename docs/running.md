# 运行LCM预训练


## Step 0：环境setup

首先我们要拉取Xiangning的分支作为origin。

```bash
git remote add origin git@github.com:XiangningLin/large_concept_model.git
```

为验证是否成功添加了新的origin，可以运行：

```bash
git remote -v
```

成功拉取远端repo之后，就可以再通过`git fetch`将代码拉取到本地。在那之后我们要做的是配置环境：

```bash
cd large_concept_model
bash INSTALL_NEW_MACHINE.sh
```

注意这里有可能会在后续训练报错说没有libsndfile，如果出现这个错误，可以新建一个conda环境，在那里面手动下载libsndfile，再activate那个环境，在该环境下进行后续数据处理和训练（注意PATH要指向这个环境）。

## Step 1：数据预处理

```bash
bash prepare_data.sh
```

数据预处理完毕之后，我们需要在datacard上面修改pretraining_data下面的parquet_path。

__TODO：__
- enable hydra风格的prepare_data和pretrain的bash命令。
- 或者，enable fire风格的prepare_data和pretrain的bash命令。