#!/usr/bin/env python3
"""
Pre-pack parquet data: combine short documents into rows of max_seq_len sentences.

Uses the same greedy packing algorithm as SentenceSSM's _pack_trajectories_pretrain.
Each original document gets an "End of text." SONAR embedding appended before packing,
serving as a document boundary marker within packed rows.

Usage:
    python scripts/pack_parquet.py \
        --source_dir /work/hdd/bfaq/jlyu3/lcm/preprocessed_data \
        --output_dir /work/hdd/bfaq/jlyu3/lcm/preprocessed_data_packed \
        --max_seq_len 128
"""

import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import glob
import fire
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from stopes.utils.arrow_utils import nested_numpy_to_pyarrow
from tqdm import tqdm


def compute_eot_embedding(suffix_text: str, device: str = "cuda:0") -> np.ndarray:
    """Compute SONAR embedding for the document boundary marker."""
    import torch
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    device = torch.device(device) if isinstance(device, str) else device
    sonar = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    )
    emb = sonar.predict([suffix_text], source_lang="eng_Latn")[0].cpu().numpy()
    return emb  # shape (1024,)


def split_oversized_doc(doc: dict, max_seq_len: int) -> list:
    """Split a document longer than max_seq_len into chunks."""
    texts = doc["text_sentences"]
    embs = doc["text_sentences_sonar_emb"]
    split_val = doc.get("split", "train")

    n = len(texts)
    if n <= max_seq_len:
        return [doc]

    chunks = []
    for start in range(0, n, max_seq_len):
        end = min(start + max_seq_len, n)
        chunks.append({
            "text_sentences": texts[start:end],
            "text_sentences_sonar_emb": embs[start:end],
            "split": split_val,
        })
    return chunks


def greedy_pack(docs: list, max_seq_len: int) -> list:
    """
    Greedy packing algorithm identical to SentenceSSM's _pack_trajectories_pretrain.

    Accumulates documents into the current row until adding the next one would
    exceed max_seq_len, then finalizes the current row and starts a new one.
    """
    packed_rows = []
    current_texts = []
    current_embs = []
    current_length = 0

    for doc in docs:
        doc_len = len(doc["text_sentences"])
        if current_length + doc_len <= max_seq_len:
            current_texts.extend(doc["text_sentences"])
            current_embs.extend(doc["text_sentences_sonar_emb"])
            current_length += doc_len
        else:
            if current_texts:
                packed_rows.append({
                    "text_sentences": current_texts,
                    "text_sentences_sonar_emb": current_embs,
                })
            current_texts = list(doc["text_sentences"])
            current_embs = list(doc["text_sentences_sonar_emb"])
            current_length = doc_len

    if current_texts:
        packed_rows.append({
            "text_sentences": current_texts,
            "text_sentences_sonar_emb": current_embs,
        })
    return packed_rows


def read_parquet_files(source_dir: str) -> list:
    """Discover all parquet files under source_dir (including shard_* subdirs)."""
    parquet_files = sorted(glob.glob(f"{source_dir}/**/data.parquet", recursive=True))
    if not parquet_files:
        parquet_files = sorted(glob.glob(f"{source_dir}/**/*.parquet", recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in: {source_dir}")
    return parquet_files


def pack_parquet(
    source_dir: str,
    output_dir: str,
    max_seq_len: int = 128,
    suffix_text: str = "End of text.",
    device: str = "cuda:0",
):
    """
    Pre-pack parquet data for LCM training.

    Reads original parquet (text_sentences, text_sentences_sonar_emb), appends an
    "End of text." SONAR embedding to each document, splits oversized documents,
    then greedily packs short documents into rows of at most max_seq_len sentences.

    Args:
        source_dir:  Directory containing original parquet files.
        output_dir:  Directory for packed parquet output.
        max_seq_len: Maximum number of sentences per packed row.
        suffix_text: Document boundary marker text.
        device:      Device for SONAR model.
    """
    print("=" * 70)
    print("Pack Parquet - Preprocessing Stage Packing")
    print("=" * 70)
    print(f"  source_dir:  {source_dir}")
    print(f"  output_dir:  {output_dir}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  suffix_text: {suffix_text!r}")
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: compute SONAR embedding for the suffix
    print("Loading SONAR model to compute suffix embedding...")
    eot_emb = compute_eot_embedding(suffix_text, device=device)
    print(f"  suffix embedding shape: {eot_emb.shape}")
    print()

    # Step 2: discover parquet files
    parquet_files = read_parquet_files(source_dir)
    print(f"Found {len(parquet_files)} parquet file(s)")

    # Step 3: read all documents, append suffix, split oversized, then pack
    total_original_docs = 0
    total_sentences_before = 0

    for split_name in ("train", "validation"):
        docs_for_split = []

        for pq_file in tqdm(parquet_files, desc=f"Reading ({split_name})"):
            table = pq.read_table(pq_file)

            has_split_col = "split" in table.column_names
            for i in range(len(table)):
                if has_split_col:
                    row_split = table["split"][i].as_py()
                    if row_split != split_name:
                        continue
                elif split_name == "validation":
                    continue

                texts = table["text_sentences"][i].as_py()
                embs_raw = table["text_sentences_sonar_emb"][i].as_py()
                embs = [np.array(e, dtype=np.float32) for e in embs_raw]

                total_original_docs += 1
                total_sentences_before += len(texts)

                texts.append(suffix_text)
                embs.append(eot_emb.astype(np.float32))

                doc = {
                    "text_sentences": texts,
                    "text_sentences_sonar_emb": embs,
                    "split": split_name,
                }

                chunks = split_oversized_doc(doc, max_seq_len)
                docs_for_split.extend(chunks)

        if not docs_for_split:
            print(f"  No documents for split={split_name}, skipping")
            continue

        print(f"\n  [{split_name}] {len(docs_for_split)} docs/chunks before packing")

        # Greedy packing
        packed_rows = greedy_pack(docs_for_split, max_seq_len)
        print(f"  [{split_name}] {len(packed_rows)} packed rows after packing")

        # Build output table
        text_sentences_list = [row["text_sentences"] for row in packed_rows]
        all_embs = [
            np.stack(row["text_sentences_sonar_emb"]) for row in packed_rows
        ]
        text_sentences_sonar_emb_pa = nested_numpy_to_pyarrow(all_embs)

        # Do NOT write split column: Hive partition (split=train, split=validation) already
        # provides it. Writing it causes schema merge errors (string vs dictionary) when
        # ParquetDataset discovers both partitions.
        out_table = pa.table({
            "text_sentences": text_sentences_list,
            "text_sentences_sonar_emb": text_sentences_sonar_emb_pa,
        })

        split_dir = output_path / f"split={split_name}"
        split_dir.mkdir(parents=True, exist_ok=True)
        out_file = split_dir / "data.parquet"
        pq.write_table(out_table, out_file, use_dictionary=False)
        print(f"  [{split_name}] Written to {out_file}")

    # Statistics (use metadata only to avoid schema merge when reading multiple files)
    total_packed_rows = 0
    for split_name in ("train", "validation"):
        split_dir = output_path / f"split={split_name}"
        if split_dir.exists():
            for f in split_dir.glob("*.parquet"):
                total_packed_rows += pq.ParquetFile(f).metadata.num_rows

    print()
    print("=" * 70)
    print("Packing complete")
    print("=" * 70)
    print(f"  Original documents:  {total_original_docs}")
    print(f"  Original sentences:  {total_sentences_before}")
    print(f"  Packed rows:         {total_packed_rows}")
    if total_packed_rows > 0:
        ratio = total_original_docs / total_packed_rows
        print(f"  Compression ratio:   {ratio:.1f}x")
    print(f"  Output directory:    {output_dir}")
    print()
    print("Next steps:")
    print("  1. Update lcm/datacards/datacards.yaml to add a packed data entry")
    print("  2. Use CLI override to point training at the packed data:")
    print('     ++trainer.training_data.0.name="fine_web_edu_packed=train"')
    print("     ++trainer.training_data.0.source_suffix_text=null")


if __name__ == "__main__":
    fire.Fire(pack_parquet)
