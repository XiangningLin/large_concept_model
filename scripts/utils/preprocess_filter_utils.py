"""
Text preprocessing and filtering utilities for data cleaning.

This module provides functions for:
1. Unicode punctuation normalization (converting smart quotes to ASCII)
2. UNK token detection and dropping (dropping texts that contain UNK tokens)
"""

from typing import Dict, Optional

import logging
import torch

logger = logging.getLogger(__name__)


# Default Unicode punctuation mapping table
# Maps Unicode smart punctuation to ASCII equivalents
DEFAULT_PUNCTUATION_MAPPING = {
    # Smart quotes
    "\u201C": '"',  # Left double quotation mark
    "\u201D": '"',  # Right double quotation mark
    "\u2018": "'",  # Left single quotation mark
    "\u2019": "'",  # Right single quotation mark
    # Dashes
    "\u2014": "-",  # Em dash
    "\u2013": "-",  # En dash
    # Ellipsis
    "\u2026": "...",  # Horizontal ellipsis
    # Angle quotes
    "\u00AB": '"',  # Left-pointing double angle quotation mark
    "\u00BB": '"',  # Right-pointing double angle quotation mark
    # Low quotes (German style)
    "\u201E": '"',  # Double low-9 quotation mark
    "\u201A": "'",  # Single low-9 quotation mark
    # Double prime
    "\u2033": '"',  # Double prime (can also be mapped to '')
    # 【 】--> [ ]
    "\u3010": "[",
    "\u3011": "]",
    # 《 》--> < >
    "\u3008": "<",
    "\u3009": ">",
}

# Create translation table for efficient character replacement
_PUNCTUATION_TRANSLATION_TABLE = str.maketrans(DEFAULT_PUNCTUATION_MAPPING)


def normalize_unicode_punctuation(
    text: str,
    custom_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Normalize Unicode punctuation to ASCII equivalents.

    This function converts smart quotes, em dashes, and other Unicode punctuation
    to their ASCII equivalents that are more likely to be recognized by tokenizers.

    Args:
        text: Input text string
        custom_mapping: Optional custom character mapping (overrides default)
                       Format: {unicode_char: ascii_replacement}

    Returns:
        Normalized text with ASCII punctuation

    Examples:
        >>> normalize_unicode_punctuation("He said "hello"")
        'He said "hello"'
        >>> normalize_unicode_punctuation("It's—no, it's–not")
        "It's-no, it's-not"
    """
    if not text:
        return text

    # Use custom mapping if provided, otherwise use default
    if custom_mapping:
        translation_table = str.maketrans(custom_mapping)
    else:
        translation_table = _PUNCTUATION_TRANSLATION_TABLE

    # Apply translation
    normalized = text.translate(translation_table)

    return normalized


def drop_text_with_unk_tokens(
    text: str,
    tokenizer,
) -> Optional[str]:
    """
    Drop text if it contains UNK tokens.

    This function tokenizes the text and checks if it contains any UNK tokens.
    If UNK tokens are found, returns None (indicating the text should be dropped).
    If no UNK tokens are found, returns the original text.

    Args:
        text: Input text string (should be after punctuation normalization)
        tokenizer: SONAR tokenizer instance (same as decoder uses)
                  Must have create_encoder() method and vocab_info.unk_idx attribute

    Returns:
        Original text if no UNK tokens found, None if UNK tokens are found (text should be dropped)

    Examples:
        >>> # Assuming tokenizer is initialized
        >>> drop_text_with_unk_tokens("Hello world", tokenizer)
        'Hello world'
        >>> drop_text_with_unk_tokens("Some text with 滇", tokenizer)
        None
    """
    if not text:
        raise ValueError("Text is empty")

    if tokenizer is None:
        raise ValueError("Tokenizer not provided")

    try:
        # Create encoder from tokenizer
        encoder = tokenizer.create_encoder(
            task="translation",
            lang="eng_Latn",
            mode="target",
            device=None,
        )

        # Get unk_id from tokenizer
        if hasattr(tokenizer, "vocab_info") and hasattr(tokenizer.vocab_info, "unk_idx"):
            unk_id = tokenizer.vocab_info.unk_idx
        else:
            unk_id = 3  # Default unk_id for SONAR tokenizer

        # Tokenize text
        token_ids_tensor = encoder(text)
        token_ids = (
            token_ids_tensor.cpu().tolist()
            if isinstance(token_ids_tensor, torch.Tensor)
            else list(token_ids_tensor)
        )

        # Flatten if nested
        if token_ids and isinstance(token_ids[0], list):
            token_ids = [tid for sublist in token_ids for tid in sublist]

        # Count UNK tokens
        unk_count = sum(1 for tid in token_ids if tid == unk_id)

        # Check if we should drop the text
        if unk_count > 0:
            logger.debug("Dropping text due to UNK tokens: %d", unk_count)
            return None

        # No UNK tokens found, return original text
        return text

    except Exception as e:
        logger.warning("Error during UNK check: %s. Returning original text.", e)
        return text
