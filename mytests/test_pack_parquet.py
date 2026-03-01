#!/usr/bin/env python3
"""
Unit tests for scripts/pack_parquet.py packing logic.

Run:
    cd large_concept_model
    python -m pytest mytests/test_pack_parquet.py -v
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.pack_parquet import split_oversized_doc, greedy_pack


SONAR_DIM = 1024
SUFFIX_TEXT = "End of text."


def make_doc(num_sentences: int, split: str = "train", prefix: str = "sent") -> dict:
    """Create a synthetic document with random SONAR embeddings."""
    texts = [f"{prefix}_{i}" for i in range(num_sentences)]
    embs = [np.random.randn(SONAR_DIM).astype(np.float32) for _ in range(num_sentences)]
    return {"text_sentences": texts, "text_sentences_sonar_emb": embs, "split": split}


def add_suffix(doc: dict) -> dict:
    """Append a fake EOT marker (mimics the SONAR embedding of 'End of text.')."""
    doc["text_sentences"] = doc["text_sentences"] + [SUFFIX_TEXT]
    doc["text_sentences_sonar_emb"] = doc["text_sentences_sonar_emb"] + [
        np.zeros(SONAR_DIM, dtype=np.float32)
    ]
    return doc


# ---------- split_oversized_doc tests ----------


def test_split_short_doc_unchanged():
    """Documents shorter than max_seq_len should not be split."""
    doc = add_suffix(make_doc(50))
    chunks = split_oversized_doc(doc, max_seq_len=128)
    assert len(chunks) == 1
    assert len(chunks[0]["text_sentences"]) == 51  # 50 + EOT


def test_split_exact_length_doc():
    """Documents exactly equal to max_seq_len should not be split."""
    doc = add_suffix(make_doc(127))  # 127 + 1 EOT = 128
    chunks = split_oversized_doc(doc, max_seq_len=128)
    assert len(chunks) == 1
    assert len(chunks[0]["text_sentences"]) == 128


def test_split_oversized_doc():
    """Documents longer than max_seq_len should be split into chunks."""
    doc = add_suffix(make_doc(200))  # 200 + 1 EOT = 201 sentences
    chunks = split_oversized_doc(doc, max_seq_len=128)
    assert len(chunks) == 2
    assert len(chunks[0]["text_sentences"]) == 128
    assert len(chunks[1]["text_sentences"]) == 73  # 201 - 128

    total = sum(len(c["text_sentences"]) for c in chunks)
    assert total == 201


def test_split_preserves_content_order():
    """Splitting should preserve sentence order."""
    doc = add_suffix(make_doc(300, prefix="s"))
    chunks = split_oversized_doc(doc, max_seq_len=128)

    reconstructed = []
    for c in chunks:
        reconstructed.extend(c["text_sentences"])
    assert reconstructed == doc["text_sentences"]


def test_split_preserves_embedding_alignment():
    """Each chunk's text_sentences and embeddings should have same length."""
    doc = add_suffix(make_doc(300))
    chunks = split_oversized_doc(doc, max_seq_len=128)
    for c in chunks:
        assert len(c["text_sentences"]) == len(c["text_sentences_sonar_emb"])


# ---------- greedy_pack tests ----------


def test_greedy_pack_combines_short_docs():
    """Two short documents should be packed into one row."""
    doc_a = add_suffix(make_doc(50))  # 51 sentences
    doc_b = add_suffix(make_doc(30))  # 31 sentences
    # 51 + 31 = 82 <= 128 -> should fit in one row
    packed = greedy_pack([doc_a, doc_b], max_seq_len=128)
    assert len(packed) == 1
    assert len(packed[0]["text_sentences"]) == 82


def test_greedy_pack_splits_when_full():
    """When adding a doc exceeds max_seq_len, start a new row."""
    doc_a = add_suffix(make_doc(60))  # 61 sentences
    doc_b = add_suffix(make_doc(60))  # 61 sentences
    doc_c = add_suffix(make_doc(60))  # 61 sentences
    # 61 + 61 = 122 <= 128 -> fit
    # 122 + 61 = 183 > 128 -> doc_c goes to new row
    packed = greedy_pack([doc_a, doc_b, doc_c], max_seq_len=128)
    assert len(packed) == 2
    assert len(packed[0]["text_sentences"]) == 122
    assert len(packed[1]["text_sentences"]) == 61


def test_greedy_pack_no_row_exceeds_max():
    """No packed row should exceed max_seq_len."""
    np.random.seed(42)
    docs = [add_suffix(make_doc(np.random.randint(5, 60))) for _ in range(50)]
    packed = greedy_pack(docs, max_seq_len=128)
    for row in packed:
        assert len(row["text_sentences"]) <= 128


def test_greedy_pack_preserves_total_sentences():
    """Total sentences across all packed rows should equal input total."""
    np.random.seed(123)
    docs = [add_suffix(make_doc(np.random.randint(1, 80))) for _ in range(30)]
    total_input = sum(len(d["text_sentences"]) for d in docs)

    packed = greedy_pack(docs, max_seq_len=128)
    total_output = sum(len(r["text_sentences"]) for r in packed)
    assert total_output == total_input


def test_greedy_pack_embedding_text_alignment():
    """Each packed row should have matching text/embedding counts."""
    docs = [add_suffix(make_doc(20)) for _ in range(10)]
    packed = greedy_pack(docs, max_seq_len=128)
    for row in packed:
        assert len(row["text_sentences"]) == len(row["text_sentences_sonar_emb"])


def test_greedy_pack_eot_positions():
    """'End of text.' should appear at original document boundaries."""
    doc_a = add_suffix(make_doc(10, prefix="a"))  # texts: a_0..a_9, EOT
    doc_b = add_suffix(make_doc(5, prefix="b"))   # texts: b_0..b_4, EOT
    packed = greedy_pack([doc_a, doc_b], max_seq_len=128)

    assert len(packed) == 1
    texts = packed[0]["text_sentences"]
    assert texts[10] == SUFFIX_TEXT  # after doc_a
    assert texts[16] == SUFFIX_TEXT  # after doc_b (at position 11+5=16)


def test_greedy_pack_single_large_doc():
    """A single document that exactly fills max_seq_len."""
    doc = add_suffix(make_doc(127))  # 128 sentences
    packed = greedy_pack([doc], max_seq_len=128)
    assert len(packed) == 1
    assert len(packed[0]["text_sentences"]) == 128


def test_greedy_pack_empty_input():
    """Packing an empty list should return an empty list."""
    packed = greedy_pack([], max_seq_len=128)
    assert packed == []


# ---------- end-to-end test with parquet I/O ----------


def test_full_pipeline_with_synthetic_parquet():
    """
    End-to-end: create synthetic parquet, run split+pack, verify output parquet.
    Does NOT require SONAR model (uses fake embeddings).
    """
    from stopes.utils.arrow_utils import nested_numpy_to_pyarrow

    np.random.seed(42)
    doc_sizes = [50, 30, 60, 200]  # Doc D is oversized
    all_texts = []
    all_embs = []
    all_splits = []

    for i, size in enumerate(doc_sizes):
        texts = [f"doc{i}_sent{j}" for j in range(size)]
        embs = np.random.randn(size, SONAR_DIM).astype(np.float32)
        all_texts.append(texts)
        all_embs.append(embs)
        all_splits.append("train")

    embs_pa = nested_numpy_to_pyarrow(all_embs)
    table = pa.table({
        "text_sentences": all_texts,
        "text_sentences_sonar_emb": embs_pa,
        "split": all_splits,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "source" / "data.parquet"
        src_file.parent.mkdir()
        pq.write_table(table, src_file)

        # Simulate the pack_parquet flow without SONAR
        eot_emb = np.zeros(SONAR_DIM, dtype=np.float32)

        parquet_files = [str(src_file)]
        docs = []
        for pq_file in parquet_files:
            t = pq.read_table(pq_file)
            for i in range(len(t)):
                texts = t["text_sentences"][i].as_py()
                embs_raw = t["text_sentences_sonar_emb"][i].as_py()
                embs = [np.array(e, dtype=np.float32) for e in embs_raw]

                texts.append(SUFFIX_TEXT)
                embs.append(eot_emb)

                doc = {
                    "text_sentences": texts,
                    "text_sentences_sonar_emb": embs,
                    "split": "train",
                }
                chunks = split_oversized_doc(doc, max_seq_len=128)
                docs.extend(chunks)

        packed = greedy_pack(docs, max_seq_len=128)

        # Verify constraints
        total_input_sentences = sum(doc_sizes) + len(doc_sizes)  # + EOT per doc
        total_packed_sentences = sum(len(r["text_sentences"]) for r in packed)
        assert total_packed_sentences == total_input_sentences

        for row in packed:
            assert len(row["text_sentences"]) <= 128
            assert len(row["text_sentences"]) == len(row["text_sentences_sonar_emb"])

        # Doc D (200 sentences + EOT = 201) gets split into chunks of 128 + 73
        # Doc A (51) + Doc B (31) = 82 -> one packed row
        # Doc C (61) + Doc D chunk2 (73) = 134 > 128 -> separate
        # So expect: [82, 61, 128, 73] -> 4 rows (order depends on accumulation)
        assert len(packed) >= 3  # at least 3 rows needed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
