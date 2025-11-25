#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Test script to visualize processed Tom Tracking data.

Usage:
    python mytests/test_tom_tracking_data.py --data_dir tom_tracking_output --num_samples 3
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pyarrow.parquet as pq


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """Print a formatted separator line."""
    if title:
        title = f" {title} "
        left_pad = (width - len(title)) // 2
        right_pad = width - len(title) - left_pad
        print(f"{char * left_pad}{title}{char * right_pad}")
    else:
        print(char * width)


def format_embedding_info(emb: np.ndarray) -> str:
    """Format embedding information for display."""
    return (
        f"Shape: {emb.shape}, "
        f"Mean: {emb.mean():.6f}, "
        f"Std: {emb.std():.6f}, "
        f"Min: {emb.min():.6f}, "
        f"Max: {emb.max():.6f}"
    )


def visualize_sample(sample: Dict[str, Any], sample_idx: int):
    """
    Visualize a single sample from the processed dataset.
    
    Args:
        sample: Dictionary containing all fields for one sample
        sample_idx: Index of the sample (for display purposes)
    """
    print_separator(f"Sample {sample_idx + 1}", char="=")
    
    # Extract fields
    instance_id = sample.get("instance_id", "N/A")
    item = sample.get("item", "N/A")
    
    print(f"\n[Metadata]")
    print(f"  Instance ID: {instance_id}")
    print(f"  Item: {item}")
    
    # ============ SOURCE (Input Prompt) ============
    print_separator("SOURCE (Input Prompt)", char="-")
    
    # Original text
    input_prompt = sample.get("input_prompt", "N/A")
    print(f"\n[Original Text] (first 500 chars):")
    print(f"  {input_prompt[:500]}...")
    print(f"  [Total length: {len(input_prompt)} characters]")
    
    # Sentences
    input_sentences = sample.get("input_prompt_sentences", [])
    print(f"\n[Sentence Splitting Results]")
    print(f"  Total sentences: {len(input_sentences)}")
    print(f"  First 3 sentences:")
    for i, sent in enumerate(input_sentences[:3]):
        print(f"    {i+1}. {sent}")
    if len(input_sentences) > 3:
        print(f"    ... ({len(input_sentences) - 3} more sentences)")
    
    # SONAR embeddings
    input_embeddings = sample.get("input_prompt_sentences_sonar_emb", None)
    if input_embeddings is not None:
        if isinstance(input_embeddings, list):
            input_embeddings = np.array(input_embeddings)
        print(f"\n[SONAR Embeddings]")
        print(f"  {format_embedding_info(input_embeddings)}")
        print(f"  First sentence embedding (first 10 dims): {input_embeddings[0, :10]}")
    
    # ============ TARGET (Reference Output) ============
    print_separator("TARGET (Reference Output)", char="-")
    
    # Original text
    reference_output = sample.get("reference_output", "N/A")
    print(f"\n[Original Text] (first 500 chars):")
    print(f"  {reference_output[:500]}...")
    print(f"  [Total length: {len(reference_output)} characters]")
    
    # Sentences
    reference_sentences = sample.get("reference_output_sentences", [])
    print(f"\n[Sentence Splitting Results]")
    print(f"  Total sentences: {len(reference_sentences)}")
    print(f"  First 3 sentences:")
    for i, sent in enumerate(reference_sentences[:3]):
        print(f"    {i+1}. {sent}")
    if len(reference_sentences) > 3:
        print(f"    ... ({len(reference_sentences) - 3} more sentences)")
    
    # SONAR embeddings
    reference_embeddings = sample.get("reference_output_sentences_sonar_emb", None)
    if reference_embeddings is not None:
        if isinstance(reference_embeddings, list):
            reference_embeddings = np.array(reference_embeddings)
        print(f"\n[SONAR Embeddings]")
        print(f"  {format_embedding_info(reference_embeddings)}")
        print(f"  First sentence embedding (first 10 dims): {reference_embeddings[0, :10]}")
    
    # ============ Data Quality Checks ============
    print_separator("Data Quality Checks", char="-")
    
    checks = []
    
    # Check 1: Sentences match embeddings shape
    if input_embeddings is not None:
        emb_match = len(input_sentences) == input_embeddings.shape[0]
        checks.append(("Source sentences match embeddings", emb_match))
    
    if reference_embeddings is not None:
        emb_match = len(reference_sentences) == reference_embeddings.shape[0]
        checks.append(("Target sentences match embeddings", emb_match))
    
    # Check 2: Embedding dimensions
    if input_embeddings is not None:
        dim_check = input_embeddings.shape[1] == 1024
        checks.append(("Source embedding dimension = 1024", dim_check))
    
    if reference_embeddings is not None:
        dim_check = reference_embeddings.shape[1] == 1024
        checks.append(("Target embedding dimension = 1024", dim_check))
    
    # Check 3: Non-empty texts
    checks.append(("Source text non-empty", len(input_prompt.strip()) > 0))
    checks.append(("Target text non-empty", len(reference_output.strip()) > 0))
    
    # Print checks
    print()
    for check_name, check_result in checks:
        status = "[PASS]" if check_result else "[FAIL]"
        print(f"  {status}: {check_name}")
    
    print()


def load_and_visualize(data_dir: Path, num_samples: int = 3):
    """
    Load parquet data and visualize samples.
    
    Args:
        data_dir: Directory containing the processed parquet files
        num_samples: Number of samples to visualize (default: 3)
    """
    print_separator("Tom Tracking Data Visualization", char="#", width=80)
    print(f"\n[Loading] Data from: {data_dir}")
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        print(f"\nPlease run prepare_tom_tracking.py first to generate the data:")
        print(f"  python scripts/prepare_tom_tracking.py --output_dir {data_dir}")
        return
    
    # Load parquet dataset
    try:
        dataset = pq.ParquetDataset(str(data_dir))
        print(f"[OK] Dataset loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        return
    
    # Display schema
    print_separator("Dataset Schema", char="-")
    print(f"\n[Schema] Available columns ({len(dataset.schema.names)}):")
    for i, col_name in enumerate(dataset.schema.names, 1):
        col_type = dataset.schema.field(col_name).type
        print(f"  {i:2d}. {col_name:40s} ({col_type})")
    
    # Load samples
    print_separator("Loading Samples", char="-")
    print(f"\n[Reading] {num_samples} samples...")
    
    try:
        table = dataset.read()
        num_rows = table.num_rows
        print(f"[OK] Total rows in dataset: {num_rows}")
        
        # Limit to available rows
        num_samples = min(num_samples, num_rows)
        
        # Convert to pandas for easier manipulation
        df = table.to_pandas()
        
        print(f"\n[Visualizing] {num_samples} samples...\n")
        
        # Visualize each sample
        for i in range(num_samples):
            sample = df.iloc[i].to_dict()
            visualize_sample(sample=sample, sample_idx=i)
        
        # Summary statistics
        print_separator("Dataset Summary Statistics", char="=")
        
        # Calculate statistics
        if "input_prompt_sentences" in df.columns:
            source_lengths = df["input_prompt_sentences"].apply(len)
            print(f"\n[Statistics] Source (Input Prompt):")
            print(f"  Sentence counts - Min: {source_lengths.min()}, "
                  f"Max: {source_lengths.max()}, "
                  f"Mean: {source_lengths.mean():.2f}, "
                  f"Median: {source_lengths.median():.2f}")
        
        if "reference_output_sentences" in df.columns:
            target_lengths = df["reference_output_sentences"].apply(len)
            print(f"\n[Statistics] Target (Reference Output):")
            print(f"  Sentence counts - Min: {target_lengths.min()}, "
                  f"Max: {target_lengths.max()}, "
                  f"Mean: {target_lengths.mean():.2f}, "
                  f"Median: {target_lengths.median():.2f}")
        
        print_separator(char="#", width=80)
        print("[COMPLETE] Visualization complete!")
        
    except Exception as e:
        print(f"[ERROR] Error reading samples: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Visualize processed Tom Tracking data")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("tom_tracking_output"),
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to visualize",
    )
    
    args = parser.parse_args()
    load_and_visualize(data_dir=args.data_dir, num_samples=args.num_samples)


if __name__ == "__main__":
    main()

