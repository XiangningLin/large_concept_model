#!/bin/bash
# 超简化测试 - 10条数据，确保能跑通
set -e

# 自动检测项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 超简化测试（10条数据）"
echo "======================================"
echo "这是最简单的测试，如果这个都不work，说明环境有问题"
echo "注意查看是否具备SSL证书"
echo ""

# 设置环境变量 - 如果未设置，使用项目目录下的临时目录
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.cache/uv_cache}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRIPT_DIR/.cache/hf_cache/datasets}"
export TMPDIR="${TMPDIR:-$SCRIPT_DIR/tmp}"

mkdir -p "$TMPDIR" logs output/test_10

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
