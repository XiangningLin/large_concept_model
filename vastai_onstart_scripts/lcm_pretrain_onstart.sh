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

# 预训练任务参数
export RECIPE="${RECIPE:-mse_370M}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/lcm/checkpoints/lcm_370m_pretrain}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lcm_370m_pretrain}"
export WANDB_PROJECT="${WANDB_PROJECT:-lcm_pretrain}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export MAX_STEPS="${MAX_STEPS:-4277}"
export NUM_GPUS="${NUM_GPUS:-8}"

# Checkpoint milestones (SentenceSSM scaling law points)
export CHECKPOINT_MILESTONES="${CHECKPOINT_MILESTONES:-[225000000,318600000,450000000,636300000,900000000,1272600000,1800000000,2545200000,3600000000,5091300000,7200000000,10182600000,14400000000]}"

# HuggingFace Hub推送配置（可选）
export HF_REPO_ID="${HF_REPO_ID:-}"
export HF_CHECKPOINT_SUBFOLDER="${HF_CHECKPOINT_SUBFOLDER:-lcm_370m_pretrain}"
export PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

# VastAI环境变量
export DEBIAN_FRONTEND=noninteractive
export GIT_TERMINAL_PROMPT=0
export MKL_THREADING_LAYER=GNU
export NCCL_TIMEOUT=1800

# 任务完成后自动关闭实例（默认 true；设 VASTAI_AUTO_SHUTDOWN=false 可禁用）
export VASTAI_AUTO_SHUTDOWN="${VASTAI_AUTO_SHUTDOWN:-true}"

# Delta传输配置（可选）
export DELTA_SSH_KEY="${DELTA_SSH_KEY:-}"
export DELTA_USER="${DELTA_USER:-jlyu3}"
export DELTA_HOST="${DELTA_HOST:-dt-login.delta.ncsa.illinois.edu}"
export DELTA_DEST="${DELTA_DEST:-/work/hdd/bfaq/jlyu3/lcm/checkpoints_vastai}"

# ========== Part 1: VastAI 适配 ==========
echo "======================================"
echo "LCM 预训练 VastAI 启动脚本"
echo "======================================"
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

# 安装wandb（如果需要）
if [ -n "$WANDB_API_KEY" ]; then
    echo "配置 wandb..."
    uv pip install --python .venv/bin/python wandb
    .venv/bin/wandb login "$WANDB_API_KEY"
fi

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

# ========== Part 5: WORK AREA（预训练任务执行）==========
echo "======================================"
echo "开始预训练任务"
echo "======================================"
echo "配置信息："
echo "  - Recipe: $RECIPE"
echo "  - 实验名称: $EXPERIMENT_NAME"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - Max steps: $MAX_STEPS"
echo "  - GPU数量: $NUM_GPUS"
echo "  - WandB项目: $WANDB_PROJECT"
echo "  - Checkpoint milestones: $CHECKPOINT_MILESTONES"
echo "======================================"

# 检查是否已有checkpoint（用于断点续训）
if [ -d "${OUTPUT_DIR}" ] && [ "$(find ${OUTPUT_DIR} -name "checkpoint_*" -o -name "milestone_*" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "⚠️ 检测到现有checkpoints，建议使用 pretrain_resume.sh 进行恢复训练"
    echo "现有checkpoints："
    find ${OUTPUT_DIR} -name "checkpoint_*" -o -name "milestone_*" 2>/dev/null | head -5
    echo ""
    if [ -t 0 ]; then
        read -p "是否继续并可能覆盖现有checkpoints？(yes/no): " continue_train
    else
        continue_train="${PRETRAIN_OVERWRITE:-yes}"
        echo "非交互模式，使用 PRETRAIN_OVERWRITE=${continue_train}"
    fi
    if [ "$continue_train" != "yes" ]; then
        echo "训练已取消"
        exit 0
    fi
fi

echo "执行预训练..."
RECIPE="$RECIPE" \
OUTPUT_DIR="$OUTPUT_DIR" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
WANDB_PROJECT="$WANDB_PROJECT" \
MAX_STEPS="$MAX_STEPS" \
NPROC="$NUM_GPUS" \
bash quick_runners/train/pretrain.sh \
    "++trainer.checkpoint_milestones=$CHECKPOINT_MILESTONES"

echo "✓ 预训练完成！"

# ========== Part 5.5: 推送到 HuggingFace Hub（可选）==========
if [ "$PUSH_TO_HUB" = "true" ] && [ -n "$HF_REPO_ID" ]; then
    echo "======================================"
    echo "推送checkpoints到HuggingFace Hub"
    echo "======================================"
    echo "目标仓库: $HF_REPO_ID"
    echo "子文件夹: $HF_CHECKPOINT_SUBFOLDER"

    # 查找所有milestone checkpoints
    MILESTONE_DIRS=$(find ${OUTPUT_DIR}/milestones -type d -name "milestone_*" 2>/dev/null || echo "")

    if [ -n "$MILESTONE_DIRS" ]; then
        echo "找到以下milestone checkpoints:"
        echo "$MILESTONE_DIRS"

        # 使用huggingface-cli上传
        for milestone_dir in $MILESTONE_DIRS; do
            milestone_name=$(basename "$milestone_dir")
            echo "上传 $milestone_name..."
            .venv/bin/huggingface-cli upload \
                "$HF_REPO_ID" \
                "$milestone_dir" \
                "${HF_CHECKPOINT_SUBFOLDER}/${milestone_name}" \
                --token "$HF_TOKEN" || {
                echo "⚠️ 上传 $milestone_name 失败，继续..."
            }
        done
        echo "✓ HuggingFace Hub 推送完成"
    else
        echo "⚠️ 未找到milestone checkpoints"
    fi
else
    echo "======================================"
    echo "跳过HuggingFace Hub推送（未配置PUSH_TO_HUB=true或HF_REPO_ID）"
    echo "======================================"
fi

# 输出checkpoint统计
echo "======================================"
echo "Checkpoint统计"
echo "======================================"
if [ -d "${OUTPUT_DIR}/milestones" ]; then
    num_milestones=$(find ${OUTPUT_DIR}/milestones -type d -name "milestone_*" 2>/dev/null | wc -l)
    echo "  - Milestones: $num_milestones"
    find ${OUTPUT_DIR}/milestones -type d -name "milestone_*" 2>/dev/null | while read dir; do
        echo "    - $(basename $dir)"
    done
fi

if [ -d "${OUTPUT_DIR}" ]; then
    num_regular=$(find ${OUTPUT_DIR} -maxdepth 1 -type d -name "checkpoint_*" 2>/dev/null | wc -l)
    echo "  - Regular checkpoints: $num_regular"
fi

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
    echo "推送checkpoints到 Delta"
    echo "======================================"
    echo "目标: ${DELTA_USER}@${DELTA_HOST}:${DELTA_DEST}"

    # 使用rsync传输checkpoints（仅传输milestones以节省时间和空间）
    if [ -d "${OUTPUT_DIR}/milestones" ]; then
        echo "传输milestone checkpoints..."
        rsync -avz --progress \
            -e "ssh -i /tmp/delta_key -o StrictHostKeyChecking=no" \
            ${OUTPUT_DIR}/milestones/ \
            ${DELTA_USER}@${DELTA_HOST}:${DELTA_DEST}/milestones/
    fi

    # 可选：传输最新的regular checkpoint
    LATEST_CKPT=$(find ${OUTPUT_DIR} -maxdepth 1 -type d -name "checkpoint_*" 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "传输最新checkpoint: $(basename $LATEST_CKPT)..."
        rsync -avz --progress \
            -e "ssh -i /tmp/delta_key -o StrictHostKeyChecking=no" \
            ${LATEST_CKPT}/ \
            ${DELTA_USER}@${DELTA_HOST}:${DELTA_DEST}/$(basename $LATEST_CKPT)/
    fi

    # 清理临时密钥
    rm /tmp/delta_key
    echo "✓ Checkpoint传输完成！"
else
    echo "======================================"
    echo "跳过Delta传输（未配置DELTA_SSH_KEY/DELTA_SSH_KEY_B64_*或DELTA_DEST）"
    echo "======================================"
fi

echo "======================================"
echo "✅ LCM预训练任务全部完成！"
echo "======================================"
echo "Checkpoint位置: ${OUTPUT_DIR}"
echo "下一步可以："
echo "  1. 运行 decay.sh 进行decay阶段训练"
echo "  2. 使用milestone checkpoints进行评估"
echo "  3. 继续在VastAI或Delta上训练更大的模型"
echo "======================================"

# ========== Part 7: 任务完成后自动关闭实例（可选）==========
# 设置 VASTAI_AUTO_SHUTDOWN=false 可禁用
if [ "${VASTAI_AUTO_SHUTDOWN:-true}" = "true" ] && [ -n "${CONTAINER_ID}" ] && [ -n "${CONTAINER_API_KEY}" ]; then
    echo "======================================"
    echo "自动关闭 VastAI 实例 (ID: ${CONTAINER_ID})"
    echo "======================================"
    resp=$(curl -s -w "\n%{http_code}" -X DELETE \
        "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/" \
        -H "Authorization: Bearer ${CONTAINER_API_KEY}" 2>/dev/null || true)
    http_code=$(echo "$resp" | tail -n1)
    if [ "$http_code" = "200" ]; then
        echo "✓ 实例已关闭"
    else
        echo "⚠️ 关闭实例失败 (HTTP $http_code)，请手动执行: vastai destroy instance ${CONTAINER_ID}"
    fi
else
    [ "${VASTAI_AUTO_SHUTDOWN:-true}" != "true" ] && echo "[Part7] 跳过自动关闭 (VASTAI_AUTO_SHUTDOWN=false)"
    [ -z "${CONTAINER_ID}" ] && echo "[Part7] 跳过自动关闭 (CONTAINER_ID 未设置)"
    [ -z "${CONTAINER_API_KEY}" ] && echo "[Part7] 跳过自动关闭 (CONTAINER_API_KEY 未设置)"
fi
