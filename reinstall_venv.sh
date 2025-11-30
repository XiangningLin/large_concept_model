#!/bin/bash
# 重新安装虚拟环境的脚本
set -e

PROJECT_ROOT="/work/nvme/bfaq/xlin5/large_concept_model"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "🔄 重新安装虚拟环境"
echo "=========================================="
echo ""
echo "⚠️  警告：这将删除现有的 .venv 目录并重新创建"
echo "项目路径: $PROJECT_ROOT"
echo ""

# 询问确认
read -p "是否继续？(yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "步骤1: 取消 VIRTUAL_ENV 环境变量..."
unset VIRTUAL_ENV
echo "✓ VIRTUAL_ENV 已取消"

echo ""
echo "步骤2: 删除旧的虚拟环境..."
if [ -d ".venv" ]; then
    echo "正在删除 .venv 目录..."
    rm -rf .venv
    echo "✓ 旧的虚拟环境已删除"
else
    echo "✓ .venv 目录不存在，跳过删除"
fi

echo ""
echo "步骤3: 检测 CUDA 版本..."
# 尝试检测 CUDA 版本
CUDA_VERSION="cu121"  # 默认值
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$CUDA_VER" ]; then
        echo "检测到 NVIDIA 驱动，使用 cu121（可根据实际情况调整）"
    fi
fi
echo "将使用 CUDA 版本: $CUDA_VERSION"
echo ""

echo "步骤4: 创建新的虚拟环境（预计5-10分钟）..."
unset VIRTUAL_ENV  # 再次确保取消
uv sync --python 3.10 --extra cpu --extra eval --extra data
echo "✓ 虚拟环境已创建"

echo ""
echo "步骤5: 安装 GPU 版本的 PyTorch..."
.venv/bin/python -c "import torch; print(f'当前PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "正在安装 PyTorch 2.5.1 (CUDA $CUDA_VERSION)..."
    uv pip install --python .venv/bin/python torch==2.5.1 \
        --extra-index-url https://download.pytorch.org/whl/$CUDA_VERSION --upgrade
}
echo "✓ PyTorch 已安装"

echo ""
echo "步骤6: 安装 GPU 版本的 fairseq2..."
.venv/bin/python -c "import fairseq2; print(f'当前fairseq2: {fairseq2.__version__}')" 2>/dev/null || {
    echo "正在安装 fairseq2 v0.3.0rc1 (CUDA $CUDA_VERSION)..."
    uv pip install --python .venv/bin/python fairseq2==v0.3.0rc1 --pre \
        --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/$CUDA_VERSION --upgrade
}
echo "✓ fairseq2 已安装"

echo ""
echo "步骤7: 安装音频处理库（libsndfile 及其依赖）..."
# 检查 .venv/lib 中是否已有 libsndfile
if [ -f ".venv/lib/libsndfile.so" ]; then
    echo "✓ 音频库已存在"
else
    echo "需要通过 conda 安装音频库..."
    
    # 检查 conda 是否可用
    if command -v conda &> /dev/null; then
        # 创建临时 conda 环境
        CONDA_ENV_NAME="lcm-helper"
        
        if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
            echo "创建 conda 环境: $CONDA_ENV_NAME"
            conda create -n $CONDA_ENV_NAME -y
        fi
        
        echo "安装 libsndfile..."
        conda install -n $CONDA_ENV_NAME -c conda-forge libsndfile==1.0.31 -y
        
        # 获取 conda 环境路径
        CONDA_LIB_PATH=$(conda info --envs | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')/lib
        
        # 复制库文件到 .venv/lib
        echo "复制音频库到 .venv/lib..."
        cp -L ${CONDA_LIB_PATH}/libsndfile.so* .venv/lib/ 2>/dev/null || true
        cp -L ${CONDA_LIB_PATH}/libFLAC.so* .venv/lib/ 2>/dev/null || true
        cp -L ${CONDA_LIB_PATH}/libvorbis*.so* .venv/lib/ 2>/dev/null || true
        cp -L ${CONDA_LIB_PATH}/libopus.so* .venv/lib/ 2>/dev/null || true
        
        echo "✓ 音频库已安装"
    else
        echo "⚠️  conda 未找到，请手动安装音频库："
        echo "   1. conda create -n lcm-helper"
        echo "   2. conda activate lcm-helper"
        echo "   3. conda install -c conda-forge libsndfile==1.0.31"
        echo "   4. cp -L \$CONDA_PREFIX/lib/{libsndfile.so*,libFLAC.so*,libvorbis*.so*,libopus.so*} .venv/lib/"
    fi
fi

echo ""
echo "步骤8: 验证安装..."
.venv/bin/python << 'PYEOF'
import sys
import torch
import fairseq2

print(f"✓ Python: {sys.version.split()[0]}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"✓ fairseq2: {fairseq2.__version__}")
PYEOF

echo ""
echo "=========================================="
echo "✅ 虚拟环境重新安装完成！"
echo "=========================================="
echo ""
echo "现在可以运行 prepare_data_and_pretrain.sh 了，不会再出现 VIRTUAL_ENV 警告。"
echo ""

