#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Test whether LCM's SONAR tokenizer recognizes smart quotes and other Unicode punctuation,
or produces UNK tokens for them.

For each toy sentence containing smart quotes:
1. Encode + decode, output token ids and decoded text
2. Check if SONAR tokenizer produces any UNK tokens in the token ids

Usage:
    cd large_concept_model && python mytests/test_sonar_smart_quotes_unk.py
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from sonar.models.sonar_text import load_sonar_tokenizer


# Toy sentences with smart quotes and other Unicode punctuation
# (same chars as in SentenceSSM's DEFAULT_PUNCTUATION_MAPPING)
TOY_SENTENCES = [
    # Smart double quotes (U+201C, U+201D)
    "He said \u201Chello\u201D to the world.",
    # Smart single quotes (U+2018, U+2019)
    "It\u2019s a beautiful day\u2014isn\u2019t it?",
    # Em dash (U+2014), En dash (U+2013)
    "The price is $100\u2013$200\u2014or so they say.",
    # Ellipsis (U+2026)
    "Wait for it\u2026",
    # Angle quotes (U+00AB, U+00BB)
    "\u00ABQuote\u00BB",
    # Mixed: ASCII quotes for comparison
    'He said "hello" to the world.',
    "It's a beautiful day—isn't it?",  # ASCII apostrophe, em dash
]


def get_unk_id(tokenizer):
    """Get UNK token id from tokenizer."""
    if hasattr(tokenizer, "vocab_info") and hasattr(tokenizer.vocab_info, "unk_idx"):
        return tokenizer.vocab_info.unk_idx
    return 3  # Default for SONAR/NLLB-style tokenizers


def get_token_strings(encoder, text):
    """Get token strings from encode_as_tokens (shows <unk> for UNK tokens)."""
    try:
        if hasattr(encoder, "encode_as_tokens"):
            tokens = encoder.encode_as_tokens(text)
            if tokens is not None:
                return [str(t) for t in tokens]
    except Exception:
        pass
    return None


def main():
    print("=" * 80)
    print("LCM SONAR Tokenizer: Smart Quotes & UNK Test")
    print("=" * 80)
    print("Tokenizer: text_sonar_basic_encoder (same as prepare_fine_web / LCM encoder)")
    print()

    # Load tokenizer (same as LCM uses for FineWeb)
    tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder", progress=False)
    encoder = tokenizer.create_encoder(
        task="translation",
        lang="eng_Latn",
        mode="target",
        device=None,
    )
    unk_id = get_unk_id(tokenizer)
    print(f"UNK token id: {unk_id}")
    print()

    for i, text in enumerate(TOY_SENTENCES):
        print("-" * 80)
        print(f"Toy sentence {i + 1}:")
        print(f"  Input:  {repr(text)}")
        print(f"  Display: {text}")

        try:
            # Encode
            token_ids_tensor = encoder(text)
            token_ids = (
                token_ids_tensor.cpu().tolist()
                if isinstance(token_ids_tensor, torch.Tensor)
                else list(token_ids_tensor)
            )
            if token_ids and isinstance(token_ids[0], list):
                token_ids = [tid for sublist in token_ids for tid in sublist]

            # Check for UNK
            unk_count = sum(1 for tid in token_ids if tid == unk_id)
            has_unk = unk_count > 0

            # Token strings (encode_as_tokens shows <unk> for UNK)
            token_strs = get_token_strings(encoder, text)

            print(f"  Token ids: {token_ids}")
            if token_strs:
                print(f"  Tokens:    {token_strs}")
            print(f"  UNK count: {unk_count} {'(HAS UNK!)' if has_unk else '(no UNK)'}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print("=" * 80)
    print("Summary: If UNK appears for smart-quote sentences, consider adding")
    print("normalize_unicode_punctuation before tokenization in prepare_fine_web.")
    print("=" * 80)


if __name__ == "__main__":
    main()
