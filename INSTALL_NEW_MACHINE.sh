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

# 步骤3: 准备项目目录
echo "3️⃣ 准备项目目录..."
# 检查是否已经在项目根目录（有 pyproject.toml 或 .venv）
if [ -f "pyproject.toml" ] || [ -d ".venv" ]; then
    echo "✓ 已在项目根目录: $(pwd)"
    PROJECT_DIR="$(pwd)"
else
    # 否则，假设 WORKSPACE 是父目录
    cd "$WORKSPACE"
    if [ ! -d "large_concept_model" ]; then
        echo "克隆仓库..."
        git clone https://github.com/facebookresearch/large_concept_model.git
    fi
    cd large_concept_model
    PROJECT_DIR="$(pwd)"
fi
cd "$PROJECT_DIR"
echo "✓ 项目目录: $PROJECT_DIR"
echo ""

# 取消 VIRTUAL_ENV 环境变量，避免路径冲突
unset VIRTUAL_ENV

# 步骤4: 创建虚拟环境
echo "4️⃣ 创建Python虚拟环境（预计5-10分钟）..."
if [ ! -d ".venv" ]; then
    uv sync --python 3.10 --extra cpu --extra eval --extra data
    echo "✓ 虚拟环境已创建"
else
    echo "⚠️  .venv 已存在"
    read -p "是否删除并重新创建虚拟环境？(yes/no，默认no): " reinstall
    if [ "$reinstall" = "yes" ]; then
        echo "删除旧的虚拟环境..."
        rm -rf .venv
        unset VIRTUAL_ENV  # 再次确保取消
        uv sync --python 3.10 --extra cpu --extra eval --extra data
        echo "✓ 虚拟环境已重新创建"
    else
        echo "✓ 使用现有虚拟环境"
    fi
fi
echo ""

# 步骤5: 安装PyTorch (GPU版本)
echo "5️⃣ 安装PyTorch (GPU版本)..."
# 检查当前 PyTorch 版本
CURRENT_TORCH=$(.venv/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -n "$CURRENT_TORCH" ]; then
    echo "当前PyTorch版本: $CURRENT_TORCH"
    # 检查是否是 CPU 版本
    if echo "$CURRENT_TORCH" | grep -q "+cpu"; then
        echo "检测到 CPU 版本，需要安装 GPU 版本..."
        echo "卸载 CPU 版本的 PyTorch..."
        uv pip uninstall --python .venv/bin/python torch torchvision torchaudio -y 2>/dev/null || true
    fi
fi

# 安装 GPU 版本的 PyTorch
echo "正在安装 PyTorch 2.5.1 (CUDA $CUDA_VERSION)..."
uv pip install --python .venv/bin/python torch==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/$CUDA_VERSION --upgrade

# 验证安装
.venv/bin/python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')" || {
    echo "⚠️  PyTorch 安装可能有问题"
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
        CONDA_ENV_NAME="lcm-helper"
        
        # 创建 conda 环境（如果不存在）
        if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
            echo "创建 conda 环境: $CONDA_ENV_NAME"
            conda create -n $CONDA_ENV_NAME -y
        else
            echo "✓ conda 环境 $CONDA_ENV_NAME 已存在"
        fi
        
        # 安装 libsndfile
        echo "安装 libsndfile..."
        conda install -n $CONDA_ENV_NAME -c conda-forge libsndfile==1.0.31 -y
        
        # 获取 conda 环境路径（多种方法尝试）
        CONDA_LIB_PATH=""
        
        # 方法1: 使用 conda info --envs（处理不同的输出格式）
        CONDA_ENV_PATH=$(conda info --envs 2>/dev/null | grep -E "^${CONDA_ENV_NAME}[[:space:]]" | awk '{print $NF}' | head -1)
        # 如果上面没找到，尝试另一种格式（可能路径在第二列）
        if [ -z "$CONDA_ENV_PATH" ]; then
            CONDA_ENV_PATH=$(conda info --envs 2>/dev/null | grep "${CONDA_ENV_NAME}" | grep -v "^#" | awk '{for(i=2;i<=NF;i++) if($i ~ /^\//) print $i}' | head -1)
        fi
        if [ -n "$CONDA_ENV_PATH" ] && [ -d "${CONDA_ENV_PATH}/lib" ]; then
            CONDA_LIB_PATH="${CONDA_ENV_PATH}/lib"
            echo "✓ 从 conda info 获取路径: $CONDA_LIB_PATH"
        fi
        
        # 方法2: 如果方法1失败，尝试常见路径
        if [ -z "$CONDA_LIB_PATH" ] || [ ! -d "$CONDA_LIB_PATH" ]; then
            USERNAME=$(whoami)
            echo "尝试常见 conda 路径..."
            for conda_base in \
                "/u/${USERNAME}/miniconda3/envs" \
                "/u/${USERNAME}/miniconda/envs" \
                "$HOME/miniconda3/envs" \
                "$HOME/miniconda/envs" \
                "/u/${USERNAME}/anaconda3/envs" \
                "$HOME/anaconda3/envs"; do
                if [ -d "${conda_base}/${CONDA_ENV_NAME}/lib" ]; then
                    CONDA_LIB_PATH="${conda_base}/${CONDA_ENV_NAME}/lib"
                    echo "✓ 找到 conda 路径: $CONDA_LIB_PATH"
                    break
                fi
            done
        fi
        
        # 方法3: 如果还是找不到，尝试从 CONDA_PREFIX 获取（如果激活了环境）
        if [ -z "$CONDA_LIB_PATH" ] || [ ! -d "$CONDA_LIB_PATH" ]; then
            if [ -n "$CONDA_PREFIX" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
                CONDA_LIB_PATH="${CONDA_PREFIX}/lib"
                echo "✓ 从 CONDA_PREFIX 获取路径: $CONDA_LIB_PATH"
            fi
        fi
        
        if [ -d "$CONDA_LIB_PATH" ]; then
            # 复制库文件到 .venv/lib
            echo "复制音频库到 .venv/lib..."
            cp -L ${CONDA_LIB_PATH}/libsndfile.so* .venv/lib/ 2>/dev/null || true
            cp -L ${CONDA_LIB_PATH}/libFLAC.so* .venv/lib/ 2>/dev/null || true
            cp -L ${CONDA_LIB_PATH}/libvorbis*.so* .venv/lib/ 2>/dev/null || true
            cp -L ${CONDA_LIB_PATH}/libopus.so* .venv/lib/ 2>/dev/null || true
            
            # 验证是否复制成功
            if [ -f ".venv/lib/libsndfile.so" ]; then
                echo "✓ 音频库已安装并复制到 .venv/lib/"
            else
                echo "⚠️  复制可能失败，请手动检查"
            fi
        else
            echo "⚠️  无法找到 conda 环境路径，请手动复制："
            echo "   cp -L \$CONDA_PREFIX/lib/{libsndfile.so*,libFLAC.so*,libvorbis*.so*,libopus.so*} .venv/lib/"
        fi
    else
        echo "⚠️  conda 未找到，请手动安装音频库："
        echo "   1. conda create -n lcm-helper"
        echo "   2. conda install -n lcm-helper -c conda-forge libsndfile==1.0.31"
        echo "   3. cp -L /u/\$(whoami)/miniconda3/envs/lcm-helper/lib/libsndfile.so* .venv/lib/"
        echo "   4. cp -L /u/\$(whoami)/miniconda3/envs/lcm-helper/lib/libFLAC.so* .venv/lib/"
        echo "   5. cp -L /u/\$(whoami)/miniconda3/envs/lcm-helper/lib/libvorbis*.so* .venv/lib/"
        echo "   6. cp -L /u/\$(whoami)/miniconda3/envs/lcm-helper/lib/libopus.so* .venv/lib/"
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
