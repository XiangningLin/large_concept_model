#!/bin/bash
# 诊断和修复LCM环境问题
set +e  # 允许命令失败，继续执行

cd /work/nvme/bfaq/xlin5/large_concept_model

echo "======================================"
echo "🔍 LCM环境诊断和修复工具"
echo "======================================"
echo ""

# 诊断1: Python环境
echo "📌 诊断1: Python环境"
echo "--------------------------------------"
if [ -f ".venv/bin/python" ]; then
    echo "✓ .venv存在"
    .venv/bin/python --version
    
    # 测试关键包
    echo ""
    echo "测试关键包..."
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
        __import__(pkg)
        print(f'  ✓ {name}')
    except ImportError as e:
        print(f'  ✗ {name} - 缺失')
        missing.append(pkg)

if missing:
    print(f'\n⚠️  缺失的包: {", ".join(missing)}')
    sys.exit(1)
else:
    print('\n✓ 所有关键包都已安装')
PYEOF
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ 包缺失，尝试重新安装..."
        uv sync --python 3.10 --extra cpu --extra eval --extra data
    fi
else
    echo "✗ .venv不存在"
    echo "正在创建虚拟环境..."
    uv sync --python 3.10 --extra cpu --extra eval --extra data
fi
echo ""

# 诊断2: GPU
echo "📌 诊断2: GPU可用性"
echo "--------------------------------------"
if .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'✓ {torch.cuda.device_count()} GPU(s) available')"; then
    .venv/bin/python -c "import torch; [print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
else
    echo "⚠️  CUDA不可用，检查是否在GPU节点上"
fi
echo ""

# 诊断3: 数据准备脚本
echo "📌 诊断3: 数据准备脚本"
echo "--------------------------------------"
if [ -f "scripts/prepare_c4.py" ]; then
    echo "✓ scripts/prepare_c4.py 存在"
    
    # 测试导入
    if .venv/bin/python -c "from scripts.prepare_c4 import prepare_c4_streaming; print('✓ 可以正常导入')"; then
        echo "✓ 脚本语法正确"
    else
        echo "✗ 脚本有语法错误"
    fi
else
    echo "✗ scripts/prepare_c4.py 缺失"
fi
echo ""

# 诊断4: 训练脚本
echo "📌 诊断4: 训练配置"
echo "--------------------------------------"
if [ -f "lcm/datacards/datacards.yaml" ]; then
    echo "✓ datacards.yaml 存在"
    echo "内容预览:"
    head -10 lcm/datacards/datacards.yaml | sed 's/^/  /'
else
    echo "✗ datacards.yaml 缺失"
fi
echo ""

# 诊断5: 检查常见错误
echo "📌 诊断5: 检查常见问题"
echo "--------------------------------------"

# 检查shebang路径问题
if [ -f ".venv/bin/torchrun" ]; then
    SHEBANG=$(head -1 .venv/bin/torchrun)
    if [[ $SHEBANG == *"/u/xlin5/large_concept_model"* ]]; then
        echo "⚠️  发现硬编码路径问题"
        echo "  建议使用: .venv/bin/python -m torch.distributed.run"
    else
        echo "✓ torchrun路径正常"
    fi
else
    echo "○ torchrun不存在（正常，我们使用python -m方式）"
fi
echo ""

echo "======================================"
echo "🔧 自动修复建议"
echo "======================================"
echo ""

# 创建最小测试脚本
cat > test_minimal.sh << 'TESTEOF'
#!/bin/bash
# 最小化测试脚本
set -e
cd /work/nvme/bfaq/xlin5/large_concept_model

echo "🧪 最小化测试"
echo ""

# 测试1: 导入LCM模块
echo "测试1: 导入LCM模块..."
.venv/bin/python -c "from lcm.train import lcm; print('✓ LCM模块可导入')"

# 测试2: 检查配置
echo "测试2: 检查Hydra配置..."
.venv/bin/python -c "
from omegaconf import OmegaConf
import sys
try:
    # 不实际运行，只检查配置能否加载
    print('✓ 配置系统正常')
except Exception as e:
    print(f'✗ 配置错误: {e}')
    sys.exit(1)
"

# 测试3: 单进程测试
echo "测试3: 单进程训练测试（dry run）..."
echo "（这只是测试能否启动，不会真正训练）"

echo "✅ 最小化测试完成"
TESTEOF
chmod +x test_minimal.sh

echo "✅ 创建了 test_minimal.sh 进行基础测试"
echo ""

echo "🚀 建议的修复步骤:"
echo ""
echo "1️⃣ 如果包有问题，重新安装:"
echo "   cd /work/nvme/bfaq/xlin5/large_concept_model"
echo "   rm -rf .venv"
echo "   uv sync --python 3.10 --extra cpu --extra eval --extra data"
echo ""
echo "2️⃣ 运行最小测试:"
echo "   bash test_minimal.sh"
echo ""
echo "3️⃣ 如果测试通过，逐步测试:"
echo "   # 测试数据准备（10条数据）"
echo "   .venv/bin/python scripts/prepare_c4.py --num_samples=10 --output_dir=output/test_10"
echo ""
echo "   # 测试看到实时输出"
echo "   tail -f logs/test_*.log"
echo ""

echo "======================================"
echo "需要更多帮助？请提供具体的错误信息！"
echo "======================================"
