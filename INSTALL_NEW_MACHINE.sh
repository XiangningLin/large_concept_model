#!/bin/bash
# 在新机器上安装LCM环境的完整脚本
set -e

echo "🚀 LCM环境安装脚本"
echo "===================="
echo ""

# 配置（根据实际情况修改这些变量）
WORKSPACE="${WORKSPACE:-$(pwd)}"
CUDA_VERSION="${CUDA_VERSION:-cu121}"  # 根据 nvidia-smi 查看的版本修改

echo "📍 配置信息:"
echo "   工作目录: $WORKSPACE"
echo "   CUDA版本: $CUDA_VERSION"
echo ""

# 步骤1: 检查CUDA
echo "1️⃣ 检查CUDA版本..."
if nvidia-smi &> /dev/null; then
    nvidia-smi | grep "CUDA Version"
    echo "✓ 检测到GPU"
else
    echo "⚠️  未检测到GPU，继续安装（可以在CPU上准备环境）"
fi
echo ""

# 步骤2: 安装uv
echo "2️⃣ 安装uv包管理器..."
if ! command -v uv &> /dev/null; then
    echo "正在安装uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    source ~/.bashrc 2>/dev/null || true
else
    echo "✓ uv已安装: $(uv --version)"
fi
echo ""

# 步骤3: 克隆仓库（如果还没有）
echo "3️⃣ 准备项目目录..."
cd "$WORKSPACE"
if [ ! -d "large_concept_model" ]; then
    echo "克隆仓库..."
    git clone https://github.com/facebookresearch/large_concept_model.git
fi
cd large_concept_model
echo "✓ 项目目录: $(pwd)"
echo ""

# 步骤4: 创建虚拟环境
echo "4️⃣ 创建Python虚拟环境（预计5-10分钟）..."
if [ ! -d ".venv" ]; then
    uv sync --python 3.10 --extra cpu --extra eval --extra data
    echo "✓ 虚拟环境已创建"
else
    echo "✓ 虚拟环境已存在"
fi
echo ""

# 步骤5: 安装PyTorch
echo "5️⃣ 安装PyTorch..."
.venv/bin/python -c "import torch; print(f'当前PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "正在安装PyTorch..."
    uv pip install --python .venv/bin/python torch==2.5.1 \
        --extra-index-url https://download.pytorch.org/whl/$CUDA_VERSION --upgrade
}
echo "✓ PyTorch已安装"
echo ""

# 步骤6: 安装fairseq2
echo "6️⃣ 安装fairseq2..."
.venv/bin/python -c "import fairseq2; print(f'当前fairseq2: {fairseq2.__version__}')" 2>/dev/null || {
    echo "正在安装fairseq2..."
    uv pip install --python .venv/bin/python fairseq2==v0.3.0rc1 --pre \
        --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/$CUDA_VERSION --upgrade
}
echo "✓ fairseq2已安装"
echo ""

# 步骤7: 安装音频处理库（libsndfile 及其依赖）
echo "7️⃣ 安装音频处理库..."
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

# 步骤8: 验证安装
echo "8️⃣ 验证安装..."
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

# 步骤9: 创建必要目录
echo "9️⃣ 创建目录结构..."
mkdir -p scripts logs output checkpoints
echo "✓ 目录已创建"
echo ""

echo "======================================"
echo "✅ 安装完成！"
echo "======================================"
echo ""
echo "📝 环境信息:"
echo "   项目路径: $(pwd)"
echo "   Python: $(.venv/bin/python --version)"
echo "   虚拟环境: .venv/"
echo ""
echo "📝 下一步:"
echo "1. 确保已复制脚本文件（prepare_data.sh, pretrain.sh等）"
echo "2. 运行测试: bash test_super_simple.sh"
echo "3. 准备数据: bash prepare_data.sh --num_gpus=2 --num_samples=2000000"
echo "4. 开始训练: bash pretrain.sh --num_gpus=2"
echo ""
