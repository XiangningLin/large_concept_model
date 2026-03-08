#!/bin/bash
set -e

# ===========Part 0: 基础Configuration（VastAI 环境变量上限 16 个，长字符串显式写在脚本内）============
# --- 脚本内显式定义（不占 VastAI env 配额）---
PYTHON_VERSION="3.10"
BRANCH_NAME="${LCM_BRANCH_NAME:-jianwen-modified}"
REPO_URL="${LCM_REPO_URL:-https://github.com/XiangningLin/large_concept_model}"
REPO_DIR="large_concept_model"

# GLOBUS_CREDS: 预打包凭证 base64，过长故写在脚本内。替换为你 prepare_gcp_creds.sh 生成的完整字符串
GLOBUS_CREDS="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"

# --- VastAI 环境变量（上限 16 个；GLOBUS_CREDS 在脚本内，不占配额）---
# 计入 16 个的典型项：GITHUB_TOKEN, GLOBUS_REFRESH_TOKEN, GLOBUS_CLIENT_ID, GLOBUS_CLIENT_SECRET,
# HF_TOKEN, LCM_BRANCH_NAME, LCM_REPO_URL, OUTPUT_DIR, START_INDEX, NUM_SAMPLES, NUM_GPUS, ...
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"
export GLOBUS_CREDS_B64="${GLOBUS_CREDS_B64:-$GLOBUS_CREDS}"
export GLOBUS_REFRESH_TOKEN="${GLOBUS_REFRESH_TOKEN:-}"
export GLOBUS_CLIENT_ID="${GLOBUS_CLIENT_ID:-}"
export GLOBUS_CLIENT_SECRET="${GLOBUS_CLIENT_SECRET:-}"

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

# Globus 相关已在 Part 0 导出（GLOBUS_CREDS_B64 来自脚本内 GLOBUS_CREDS）

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

# ========== Part 2.5: Globus GCP 安装（可选）==========
# 仅当 GLOBUS_CREDS_B64 已设置（脚本内 GLOBUS_CREDS 或 VastAI env）且 globus_connection 存在时执行
GLOBUS_DIR="$(pwd)/vastai_onstart_scripts/globus_connection"
if [ -n "${GLOBUS_CREDS_B64}" ] && [ -f "${GLOBUS_DIR}/install_gcp.sh" ]; then
    echo "======================================"
    echo "安装 Globus Connect Personal"
    echo "======================================"
    bash "${GLOBUS_DIR}/install_gcp.sh" || echo "⚠️ Globus GCP 安装失败，将跳过传输"
else
    GLOBUS_DIR=""
fi

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

# ========== Part 6: Globus 传输到 Delta（可选）==========
# 仅当 GLOBUS_REFRESH_TOKEN、GLOBUS_CLIENT_ID 已设置且 globus_connection 模块存在时执行
if [ -n "${GLOBUS_REFRESH_TOKEN}" ] && [ -n "${GLOBUS_CLIENT_ID}" ] && [ -n "${GLOBUS_DIR}" ] && [ -f "${GLOBUS_DIR}/transfer.py" ]; then
    echo "======================================"
    echo "通过 Globus 传输到 Delta"
    echo "======================================"
    uv pip install --python .venv/bin/python -q globus-sdk 2>/dev/null || .venv/bin/pip install -q globus-sdk 2>/dev/null || true
    export GLOBUS_SOURCE_PATH="${OUTPUT_DIR}"
    # GCP 以 globus 用户运行，凭证在 /home/globus/.globusonline；设置 HOME 以便 transfer.py 找到 endpoint
    chmod -R a+rX "${OUTPUT_DIR}" 2>/dev/null || true
    HOME=/home/globus .venv/bin/python "${GLOBUS_DIR}/transfer.py" || echo "⚠️ Globus 传输提交失败"
fi

# ========== Part 7: 原 Delta SSH 推送（已弃用，保留注释）==========
# # 支持 DELTA_SSH_KEY（单变量）或 DELTA_SSH_KEY_B64_1/2/3（多段 Base64，VastAI 256 字符限制）
# if [ -n "${DELTA_DEST}" ]; then
#     if [ -n "${DELTA_SSH_KEY}" ]; then
#         echo "${DELTA_SSH_KEY}" > /tmp/delta_key
#     elif [ -n "${DELTA_SSH_KEY_B64_1}" ]; then
#         echo "${DELTA_SSH_KEY_B64_1}${DELTA_SSH_KEY_B64_2}${DELTA_SSH_KEY_B64_3}" | base64 -d > /tmp/delta_key
#     fi
# fi

# if [ -f /tmp/delta_key ] && [ -n "${DELTA_DEST}" ]; then
#     chmod 600 /tmp/delta_key
#     echo "======================================"
#     echo "推送数据到 Delta"
#     echo "======================================"
#     echo "目标: ${DELTA_USER}@${DELTA_HOST}:${DELTA_DEST}"

#     # 使用rsync传输数据
#     rsync -avz --progress \
#         -e "ssh -i /tmp/delta_key -o StrictHostKeyChecking=no" \
#         ${OUTPUT_DIR}/ \
#         ${DELTA_USER}@${DELTA_HOST}:${DELTA_DEST}/

#     # 清理临时密钥
#     rm /tmp/delta_key
#     echo "✓ 数据传输完成！"
# else
#     echo "======================================"
#     echo "跳过数据传输（未配置DELTA_SSH_KEY/DELTA_SSH_KEY_B64_*或DELTA_DEST）"
#     echo "======================================"
# fi

echo "======================================"
echo "✅ LCM预处理任务全部完成！"
echo "======================================"
echo "输出位置: ${OUTPUT_DIR}"
echo "下一步可以："
echo "  1. 在Delta上使用这些数据进行训练"
echo "  2. 或者在VastAI上继续运行 lcm_pretrain_onstart.sh"
echo "======================================"
