#!/bin/bash
# 超简化测试 - 10条数据，确保能跑通
set -e

cd /work/nvme/bfaq/xlin5/large_concept_model

echo "🧪 超简化测试（10条数据）"
echo "======================================"
echo "这是最简单的测试，如果这个都不work，说明环境有问题"
echo ""

# 设置环境变量
export UV_CACHE_DIR=/work/nvme/bfaq/xlin5/uv_cache
export HF_HOME=/work/nvme/bfaq/xlin5/hf_cache
export HF_DATASETS_CACHE=/work/nvme/bfaq/xlin5/hf_cache/datasets
export TMPDIR=/work/nvme/bfaq/xlin5/tmp
mkdir -p $TMPDIR logs output/test_10

echo "步骤1: 测试Python环境..."
.venv/bin/python --version
.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "步骤2: 准备10条数据（预计2-3分钟）..."
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/prepare_c4.py \
    --output_dir=output/test_10 \
    --num_samples=10 \
    --batch_size=5

echo ""
echo "步骤3: 检查输出..."
if [ -f "output/test_10/data.parquet" ]; then
    ls -lh output/test_10/data.parquet
    echo "✅ 数据准备成功！"
else
    echo "❌ 数据准备失败"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ 超简化测试完成！"
echo ""
echo "如果这个测试通过，说明环境正常。"
echo "可以继续测试更复杂的场景。"
echo "======================================"
