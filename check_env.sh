#!/bin/bash
# 快速检查环境是否安装正确

echo "=========================================="
echo "🔍 检查 LCM 环境安装状态"
echo "=========================================="
echo ""

# 检查1: 虚拟环境是否存在
echo "1️⃣ 检查虚拟环境..."
if [ -f ".venv/bin/python" ]; then
    echo "✓ .venv 存在"
    .venv/bin/python --version
else
    echo "❌ .venv 不存在"
    exit 1
fi
echo ""

# 检查2: 关键 Python 包
echo "2️⃣ 检查关键 Python 包..."
.venv/bin/python << 'PYEOF'
import sys
packages = {
    'torch': 'PyTorch',
    'fairseq2': 'fairseq2',
    'datasets': 'HuggingFace datasets',
    'sonar': 'SONAR',
    'stopes': 'stopes',
    'fire': 'fire',
    'tqdm': 'tqdm',
    'pyarrow': 'pyarrow',
}

missing = []
for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {name}: {version}')
    except ImportError as e:
        print(f'  ❌ {name} - 缺失')
        missing.append(pkg)

if missing:
    print(f'\n⚠️  缺失的包: {", ".join(missing)}')
    sys.exit(1)
else:
    print('\n✓ 所有关键包都已安装')
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 包检查失败"
    exit 1
fi
echo ""

# 检查3: libsndfile 库
echo "3️⃣ 检查音频库 (libsndfile)..."
if [ -f ".venv/lib/libsndfile.so" ]; then
    echo "✓ libsndfile.so 存在"
    ls -lh .venv/lib/libsndfile.so* 2>/dev/null | head -3
else
    echo "⚠️  libsndfile.so 不存在（fairseq2 可能需要它）"
    echo "   如果遇到 fairseq2 导入错误，需要安装 libsndfile"
fi
echo ""

# 检查4: GPU 支持
echo "4️⃣ 检查 GPU 支持..."
.venv/bin/python << 'PYEOF'
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠️  CUDA 不可用（可能是 CPU 版本或不在 GPU 节点上）")
PYEOF
echo ""

# 检查5: 测试 fairseq2 导入（这是最容易出错的地方）
echo "5️⃣ 测试 fairseq2 导入..."
.venv/bin/python -c "import fairseq2; print('✓ fairseq2 可以正常导入')" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ fairseq2 导入成功"
else
    echo "❌ fairseq2 导入失败"
    echo "   这通常是因为缺少 libsndfile 库"
    exit 1
fi
echo ""

# 检查6: 测试 LCM 模块导入
echo "6️⃣ 测试 LCM 模块导入..."
.venv/bin/python -c "from lcm.train import lcm; print('✓ LCM 模块可以正常导入')" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ LCM 模块导入成功"
else
    echo "⚠️  LCM 模块导入失败（可能不影响基本使用）"
fi
echo ""

echo "=========================================="
echo "✅ 环境检查完成！"
echo "=========================================="
echo ""
echo "如果所有检查都通过，环境已安装完成。"
echo "可以运行: bash prepare_data_and_pretrain.sh"
echo ""





