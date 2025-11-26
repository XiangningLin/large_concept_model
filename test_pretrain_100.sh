#!/bin/bash
# 测试预训练 - 使用100条数据
set -e

cd /work/nvme/bfaq/xlin5/large_concept_model

echo "🧪 测试LCM 780M预训练（100条数据）"
echo "==========================================="
echo "数据: output/c4_100_stream (100条样本)"
echo "模型: 780M参数"
echo "GPU: 2张"
echo "训练步数: 20步（测试用）"
echo "输出: checkpoints/test_lcm_780m_100"
echo "==========================================="

# 设置环境变量
export UV_CACHE_DIR=/work/nvme/bfaq/xlin5/uv_cache
export TMPDIR=/work/nvme/bfaq/xlin5/tmp
mkdir -p $TMPDIR

# 创建输出目录
mkdir -p checkpoints/test_lcm_780m_100

# 2张GPU测试训练
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=2 \
    -m lcm.train \
    launcher=standalone \
    +pretrain=mse_780m \
    ++trainer.output_dir=checkpoints/test_lcm_780m_100 \
    ++trainer.experiment_name=test_100_samples \
    ++trainer.data_loading_config.max_tokens=500 \
    ++trainer.max_steps=20 \
    ++trainer.checkpoint_every_n_steps=10 \
    ++trainer.publish_metrics_every_n_steps=1 \
    ++trainer.validate_every_n_steps=10 \
    ++trainer.use_fsdp=true

echo ""
echo "✅ 测试训练完成！"
echo "📁 Checkpoints: checkpoints/test_lcm_780m_100/"
