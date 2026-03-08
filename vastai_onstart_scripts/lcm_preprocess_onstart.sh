#!/bin/bash
set -e

# ===========Part 0: 基础Configuration============
PYTHON_VERSION="3.10"
BRANCH_NAME="${LCM_BRANCH_NAME:-jianwen-modified}"
REPO_URL="${LCM_REPO_URL:-https://github.com/XiangningLin/large_concept_model}"
REPO_DIR="large_concept_model"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"  # 私有 fork 需要

# LCM特有环境变量
export HF_TOKEN="${HF_TOKEN:-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}"
export UV_CACHE_DIR="/workspace/lcm/uv_cache"
export HF_HOME="/workspace/lcm/hf_cache"
export HF_DATASETS_CACHE="/workspace/lcm/hf_cache/datasets"
export TMPDIR="/workspace/lcm/tmp"

# 预处理任务参数
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/lcm/preprocessed_data}"
export START_INDEX="${START_INDEX:-0}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}" # 1000000
export NUM_GPUS="${NUM_GPUS:-1}" # 8
export BATCH_SIZE="${BATCH_SIZE:-64}"
export SONAR_BATCH_SIZE="${SONAR_BATCH_SIZE:-128}"

# VastAI环境变量
export DEBIAN_FRONTEND=noninteractive
export GIT_TERMINAL_PROMPT=0
export MKL_THREADING_LAYER=GNU

# Delta传输配置（可选）
export DELTA_SSH_KEY="${DELTA_SSH_KEY:-}"
export DELTA_USER="${DELTA_USER:-jlyu3}"
export DELTA_HOST="${DELTA_HOST:-dt-login.delta.ncsa.illinois.edu}"
export DELTA_DEST="${DELTA_DEST:-/work/hdd/bfaq/jlyu3/lcm/preprocessed_data}"

# ========== Part 1: VastAI 适配 ==========
echo "======================================"
echo "LCM 预处理 VastAI 启动脚本"
echo "======================================"

# 提前创建 LCM 目录（uv 安装脚本会用到 TMPDIR）
mkdir -p ${OUTPUT_DIR} ${UV_CACHE_DIR} ${HF_HOME} ${HF_DATASETS_CACHE} ${TMPDIR}

echo "安装系统依赖..."
apt-get update -qq && apt-get install -y build-essential libsndfile1 nano wget git curl rsync

# 安装uv（LCM特有）
if ! command -v uv &> /dev/null; then
    echo "正在安装uv包管理器..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

# ========== Part 2: UV 虚拟环境创建 ==========
echo "======================================"
echo "创建UV虚拟环境"
echo "======================================"
cd /workspace
if [ ! -d "$REPO_DIR" ]; then
    echo "克隆仓库: $REPO_URL (分支: $BRANCH_NAME)"
    if [ -n "$GITHUB_TOKEN" ]; then
        git clone --branch $BRANCH_NAME "https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
    else
        git clone --branch $BRANCH_NAME "$REPO_URL"
    fi
fi
cd "$REPO_DIR"

# 创建uv虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "创建uv虚拟环境（Python $PYTHON_VERSION）..."
    uv sync --python $PYTHON_VERSION --extra cpu --extra eval --extra data
    echo "✓ 虚拟环境已创建"
else
    echo "✓ .venv 已存在，跳过创建"
fi

# 设置环境变量（类似conda activate）
export VIRTUAL_ENV="$(pwd)/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONPATH

# ========== Part 3: 仓库克隆 ==========
# 已在 Part 2 完成

# ========== Part 4: 依赖安装 ==========
echo "======================================"
echo "安装依赖"
echo "======================================"

# 检查当前 PyTorch 版本
CURRENT_TORCH=$(.venv/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -n "$CURRENT_TORCH" ]; then
    echo "当前PyTorch版本: $CURRENT_TORCH"
    if echo "$CURRENT_TORCH" | grep -q "+cpu"; then
        echo "检测到 CPU 版本，需要安装 GPU 版本..."
        uv pip uninstall --python .venv/bin/python torch torchvision torchaudio -y 2>/dev/null || true
    fi
fi

# 安装PyTorch (GPU版本)
echo "安装 PyTorch 2.5.1 (CUDA 12.4)..."
uv pip install --python .venv/bin/python torch==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cu124 --upgrade

# 验证PyTorch安装
.venv/bin/python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')" || {
    echo "⚠️ PyTorch 安装可能有问题"
}

# 安装fairseq2 (LCM使用0.3.0rc1)
# 注意：libsndfile1 已在 Part 1 通过 apt 安装，fairseq2n 依赖它
echo "安装 fairseq2 0.3.0rc1..."
uv pip install --python .venv/bin/python fairseq2==v0.3.0rc1 --pre \
    --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu124 --upgrade

.venv/bin/python -c "import fairseq2; print(f'✓ fairseq2: {fairseq2.__version__}')" || {
    echo "⚠️ fairseq2 安装可能有问题（需确保 libsndfile1 已通过 apt 安装）"
}

# 创建必要目录
mkdir -p ${OUTPUT_DIR} ${UV_CACHE_DIR} ${HF_HOME} ${TMPDIR}

# 验证安装
echo "======================================"
echo "验证安装"
echo "======================================"
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

# ========== Part 5: WORK AREA（预处理任务执行）==========
echo "======================================"
echo "开始预处理任务"
echo "======================================"
echo "配置信息："
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 起始索引: $START_INDEX"
echo "  - 每卡样本数: $NUM_SAMPLES"
echo "  - GPU数量: $NUM_GPUS"
echo "  - HF Token: ${HF_TOKEN:0:10}..."
echo "======================================"

# 检查是否已有数据
if [ -d "${OUTPUT_DIR}/rank_0" ] && [ "$(find ${OUTPUT_DIR}/rank_0 -name "*.parquet" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✓ 数据已存在于 ${OUTPUT_DIR}/rank_0，跳过预处理..."
else
    echo "执行多GPU流式预处理..."
    OUTPUT_DIR="$OUTPUT_DIR" \
    START_INDEX="$START_INDEX" \
    NUM_SAMPLES="$NUM_SAMPLES" \
    NUM_GPUS="$NUM_GPUS" \
    bash quick_runners/preprocess/prep_fineweb_streaming_multigpu.sh \
        --batch_size="${BATCH_SIZE}" \
        --sonar_batch_size="${SONAR_BATCH_SIZE}"

    echo "✓ 预处理完成！"
fi

echo "======================================"
echo "输出文件统计"
echo "======================================"
for rank_dir in ${OUTPUT_DIR}/rank_*; do
    if [ -d "$rank_dir" ]; then
        num_files=$(find "$rank_dir" -name "*.parquet" 2>/dev/null | wc -l)
        echo "  - $(basename $rank_dir): $num_files parquet files"
    fi
done

# ========== Part 6: 数据推送到 Delta（可选）==========
# 支持 DELTA_SSH_KEY（单变量）或 DELTA_SSH_KEY_B64_1/2/3（多段 Base64，VastAI 256 字符限制）
if [ -n "${DELTA_DEST}" ]; then
    if [ -n "${DELTA_SSH_KEY}" ]; then
        echo "${DELTA_SSH_KEY}" > /tmp/delta_key
    elif [ -n "${DELTA_SSH_KEY_B64_1}" ]; then
        echo "${DELTA_SSH_KEY_B64_1}${DELTA_SSH_KEY_B64_2}${DELTA_SSH_KEY_B64_3}" | base64 -d > /tmp/delta_key
    fi
fi

if [ -f /tmp/delta_key ] && [ -n "${DELTA_DEST}" ]; then
    chmod 600 /tmp/delta_key
    echo "======================================"
    echo "推送数据到 Delta"
    echo "======================================"
    echo "目标: ${DELTA_USER}@${DELTA_HOST}:${DELTA_DEST}"

    # 使用rsync传输数据
    rsync -avz --progress \
        -e "ssh -i /tmp/delta_key -o StrictHostKeyChecking=no" \
        ${OUTPUT_DIR}/ \
        ${DELTA_USER}@${DELTA_HOST}:${DELTA_DEST}/

    # 清理临时密钥
    rm /tmp/delta_key
    echo "✓ 数据传输完成！"
else
    echo "======================================"
    echo "跳过数据传输（未配置DELTA_SSH_KEY/DELTA_SSH_KEY_B64_*或DELTA_DEST）"
    echo "======================================"
fi

echo "======================================"
echo "✅ LCM预处理任务全部完成！"
echo "======================================"
echo "输出位置: ${OUTPUT_DIR}"
echo "下一步可以："
echo "  1. 在Delta上使用这些数据进行训练"
echo "  2. 或者在VastAI上继续运行 lcm_pretrain_onstart.sh"
echo "======================================"
