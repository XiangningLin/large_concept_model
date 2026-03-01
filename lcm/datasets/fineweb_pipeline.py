# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FineWeb preprocessing pipeline for stopes-based multi-GPU data preparation.

Replicates the exact preprocessing logic of scripts/prepare_fine_web.py
as a stopes BatchMapper, enabling parallel execution via stopes_data_mapper.

Pipeline steps per batch:
  1. Unicode punctuation normalization
  2. Sentence splitting (sentence_splitter package, rule-based)
  3. Truncation to max_sentence_length
  4. UNK token filtering (optional)
  5. SONAR embedding
  6. Deterministic train/validation split assignment
"""

import gc
import logging
import typing as tp
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from sentence_splitter import SentenceSplitter as RuleSentenceSplitter
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from stopes.modules.partitioned_data_mapper import BatchMapper
from stopes.utils.arrow_utils import nested_numpy_to_pyarrow

from scripts.utils.preprocess_filter_utils import (
    drop_text_with_unk_tokens,
    normalize_unicode_punctuation,
)

logger = logging.getLogger(__name__)


@dataclass
class FineWebPipelineConfig:
    max_sentence_length: int = 256
    train_ratio: float = 0.8
    seed: int = 42
    enable_unk_filter: bool = True
    sonar_batch_size: int = 10
    language: str = "en"
    source_lang: str = "eng_Latn"


class FineWebPipeline(BatchMapper):
    """
    Stopes BatchMapper that processes raw FineWeb text into
    sentence-split, SONAR-embedded parquet rows.

    Input pa.Table must contain a ``text`` column (string).
    Output pa.Table contains:
      - text_sentences: list<string>
      - text_sentences_sonar_emb: list<fixed_size_list<float32>[1024]>
      - split: string ("train" or "validation")
      - url, timestamp (passed through if present)
    """

    def __init__(self, config: FineWebPipelineConfig):
        super().__init__(config)
        self.config = config
        self._splitter: tp.Optional[RuleSentenceSplitter] = None
        self._sonar: tp.Optional[TextToEmbeddingModelPipeline] = None
        self._tokenizer = None
        self._tokenizer_loaded = False

    def _get_splitter(self) -> RuleSentenceSplitter:
        if self._splitter is None:
            self._splitter = RuleSentenceSplitter(language=self.config.language)
        return self._splitter

    def _get_sonar(self) -> TextToEmbeddingModelPipeline:
        if self._sonar is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._sonar = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=device,
            )
        return self._sonar

    def _get_tokenizer(self):
        if not self._tokenizer_loaded:
            self._tokenizer_loaded = True
            if self.config.enable_unk_filter:
                try:
                    from sonar.models.sonar_text import load_sonar_tokenizer

                    self._tokenizer = load_sonar_tokenizer(
                        "text_sonar_basic_encoder", progress=False
                    )
                except Exception as e:
                    logger.warning("Could not load SONAR tokenizer for UNK filter: %s", e)
                    self._tokenizer = None
        return self._tokenizer

    def _split_and_filter_one(self, text: str) -> tp.Optional[tp.List[str]]:
        """Normalize, split, truncate, and UNK-filter a single document."""
        text = text.strip()
        if not text:
            return None

        text = normalize_unicode_punctuation(text)
        if not text.strip():
            return None

        splitter = self._get_splitter()
        sentences = splitter.split(text)
        if not sentences:
            return None

        sentences = [s.strip()[: self.config.max_sentence_length] for s in sentences if s.strip()]

        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            sentences = [
                s for s in sentences if s and drop_text_with_unk_tokens(s, tokenizer) is not None
            ]

        return sentences if sentences else None

    def _assign_split(self, index: int) -> str:
        """Deterministic train/validation assignment using hash."""
        if hash((index, self.config.seed)) % 100 < int(self.config.train_ratio * 100):
            return "train"
        return "validation"

    def __call__(self, batch: tp.Optional[pa.Table]) -> tp.Optional[pa.Table]:
        if batch is None or len(batch) == 0:
            return None

        texts = batch.column("text").to_pylist()

        all_sentences: tp.List[tp.List[str]] = []
        all_embeddings: tp.List[np.ndarray] = []
        all_urls: tp.List[str] = []
        all_timestamps: tp.List[str] = []
        all_splits: tp.List[str] = []

        has_url = "url" in batch.column_names
        has_timestamp = "timestamp" in batch.column_names
        urls = batch.column("url").to_pylist() if has_url else [""] * len(texts)
        timestamps = batch.column("timestamp").to_pylist() if has_timestamp else [""] * len(texts)

        sonar = self._get_sonar()

        for i, text in enumerate(texts):
            sentences = self._split_and_filter_one(text)
            if sentences is None:
                continue

            try:
                embeddings = sonar.predict(sentences, source_lang=self.config.source_lang)
                emb_np = embeddings.cpu().numpy()
            except Exception as e:
                logger.warning("SONAR encoding failed for row %d: %s", i, e)
                continue

            all_sentences.append(sentences)
            all_embeddings.append(emb_np)
            all_urls.append(urls[i] if urls[i] else "")
            all_timestamps.append(timestamps[i] if timestamps[i] else "")
            all_splits.append(self._assign_split(i))

        if not all_sentences:
            return None

        emb_pa = nested_numpy_to_pyarrow(all_embeddings)

        result = pa.table(
            {
                "text_sentences": all_sentences,
                "text_sentences_sonar_emb": emb_pa,
                "url": all_urls,
                "timestamp": all_timestamps,
                "split": all_splits,
            }
        )

        gc.collect()
        return result

    def clear_memory(self) -> None:
        if self._sonar is not None:
            del self._sonar
            self._sonar = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            self._tokenizer_loaded = False
        self._splitter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
