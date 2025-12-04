# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Datacard Management Script
用于生成和更新 datacard 条目到 lcm/datacards/datacards.yaml
"""

import re
from pathlib import Path
from typing import Optional


def generate_datacard_entry(
    output_dir: str,
    dataset_name: str,
    cluster_name: str = "s3",
    source_column: str = "text_sentences_sonar_emb",
    source_text_column: str = "text_sentences",
    target_column: Optional[str] = None,
    target_text_column: Optional[str] = None,
    add_split_column: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> str:
    """生成 datacard 条目字符串
    
    参数:
        output_dir: 输出目录路径
        dataset_name: 数据集名称
        cluster_name: 集群名称（s3 或 local），默认 "s3"
        source_column: source embedding 列名
        source_text_column: source text 列名
        target_column: target embedding 列名（可选，用于 finetuning）
        target_text_column: target text 列名（可选，用于 finetuning）
        add_split_column: 是否添加 split 列
        train_ratio: 训练集比例
        seed: 随机种子
    """
    # 判断路径类型（如果 output_dir 是绝对路径，使用 local；否则使用 cluster_name）
    is_absolute = Path(output_dir).is_absolute()
    path_key = "local" if is_absolute else cluster_name
    
    datacard = f"""name: "{dataset_name}"
parquet_path:
  {path_key}: "{output_dir}"
source_column: "{source_column}"
source_text_column: "{source_text_column}"
"""
    
    # 添加 target 列（如果提供）
    if target_column and target_text_column:
        datacard += f"""target_column: "{target_column}"
target_text_column: "{target_text_column}"
"""
    
    # 添加 partition columns 注释（如果启用）
    if add_split_column:
        train_pct = int(train_ratio * 100)
        val_pct = int((1 - train_ratio) * 100)
        datacard += f"""# partition columns:
# "split" (train, validation) - {train_pct}% train, {val_pct}% validation with seed={seed}
"""
    
    return datacard


def update_datacards_file(
    datacard_entry: str,
    dataset_name: str,
    datacards_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """更新或追加 datacard 到 datacards.yaml 文件
    
    参数:
        datacard_entry: datacard 条目字符串
        dataset_name: 数据集名称（用于查找已存在的条目）
        datacards_path: datacards.yaml 文件路径，如果为 None 则自动查找
    
    Returns:
        (updated: bool, action: str) - updated 表示是否更新了已存在的条目，action 是 "更新" 或 "追加"
    """
    # 自动查找 datacards.yaml 路径
    if datacards_path is None:
        # 假设脚本在 scripts/ 目录下
        script_dir = Path(__file__).parent
        datacards_path = script_dir.parent / "lcm" / "datacards" / "datacards.yaml"
    
    datacards_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取现有文件
    if datacards_path.exists():
        content = datacards_path.read_text(encoding="utf-8")
    else:
        content = ""
    
    # 按 --- 分割文档
    documents = [doc.strip() for doc in re.split(r'^---\s*$', content, flags=re.MULTILINE) if doc.strip()]
    
    # 查找是否已存在相同 name 的文档
    name_pattern = re.compile(r'^name:\s*["\']?([^"\']+)["\']?', re.MULTILINE)
    found_index = -1
    
    for i, doc in enumerate(documents):
        match = name_pattern.search(doc)
        if match and match.group(1) == dataset_name:
            found_index = i
            break
    
    # 更新或追加
    if found_index >= 0:
        documents[found_index] = datacard_entry.strip()
        action = "更新"
    else:
        documents.append(datacard_entry.strip())
        action = "追加"
    
    # 重新组合文件内容（用 --- 分隔）
    new_content = "\n---\n".join(documents)
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    
    # 写入文件
    datacards_path.write_text(new_content, encoding="utf-8")
    
    return found_index >= 0, action


def update_datacard(
    output_dir: str,
    dataset_name: str,
    cluster_name: str = "s3",
    source_column: str = "text_sentences_sonar_emb",
    source_text_column: str = "text_sentences",
    target_column: Optional[str] = None,
    target_text_column: Optional[str] = None,
    add_split_column: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
    datacards_path: Optional[str] = None,
    dry_run: bool = False,
):
    """生成并更新 datacard 到 datacards.yaml（主函数，可通过 fire 调用）
    
    参数:
        output_dir: 输出目录路径
        dataset_name: 数据集名称
        cluster_name: 集群名称（s3 或 local），默认 "s3"
        source_column: source embedding 列名
        source_text_column: source text 列名
        target_column: target embedding 列名（可选）
        target_text_column: target text 列名（可选）
        add_split_column: 是否添加 split 列
        train_ratio: 训练集比例
        seed: 随机种子
        datacards_path: datacards.yaml 文件路径（可选）
        dry_run: 如果为 True，只打印不写入文件
    
    使用示例:
        # 基本用法
        python scripts/update_datacards.py \
            --output_dir=output/fine_web \
            --dataset_name=fine_web_edu
        
        # 完整参数
        python scripts/update_datacards.py \
            --output_dir=output/fine_web \
            --dataset_name=fine_web_edu \
            --cluster_name=s3 \
            --add_split_column=True \
            --train_ratio=0.9 \
            --seed=123
    """
    print("=" * 80)
    print("📋 生成 Datacard 条目...")
    print("=" * 80)
    
    # 生成 datacard 条目
    datacard_entry = generate_datacard_entry(
        output_dir=output_dir,
        dataset_name=dataset_name,
        cluster_name=cluster_name,
        source_column=source_column,
        source_text_column=source_text_column,
        target_column=target_column,
        target_text_column=target_text_column,
        add_split_column=add_split_column,
        train_ratio=train_ratio,
        seed=seed,
    )
    
    print("\n生成的 datacard 内容:")
    print("-" * 80)
    print(datacard_entry)
    print("-" * 80)
    
    if dry_run:
        print("\n🔍 Dry run 模式：未实际更新文件")
        return
    
    # 更新 datacards.yaml
    datacards_path_obj = Path(datacards_path) if datacards_path else None
    updated, action = update_datacards_file(datacard_entry, dataset_name, datacards_path_obj)
    
    print(f"\n✅ 已{action} datacard 到: {datacards_path_obj or 'lcm/datacards/datacards.yaml'}")
    if updated:
        print(f"   (替换了已存在的 '{dataset_name}' 条目)")
    else:
        print(f"   (新增了 '{dataset_name}' 条目)")
    print("=" * 80)


if __name__ == "__main__":
    import fire
    fire.Fire(update_datacard)

