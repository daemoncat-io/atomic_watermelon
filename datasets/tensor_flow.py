"""
TFDS Corpus Builder
===================
Extracts raw text from tensorflow_datasets catalog entries,
deduplicates, filters, and writes a single train_corpus.txt.

Plug this output into your existing tokenizer/training pipeline.

Usage:
    python tfds_corpus_builder.py

    Or import and call directly:
        from tfds_corpus_builder import extract_text_from_tfds, build_corpus
"""

from typing import Optional
from pathlib import Path

import tensorflow as tfds
import hashlib
import json
import os


def extract_text_from_tfds(
    dataset_name: str,
    split: str = "train",
    text_keys: Optional[list[str]] = None,
    output_dir: str = "./corpus_raw",
) -> Path:
    """
    Pull raw text out of a TFDS dataset and dump to disk as JSONL.
    Tensorflow is quarantined to this function — nothing leaks out.

    Args:
        dataset_name: TFDS dataset identifier (e.g. 'scientific_papers', 'wiki40b/en')
        split: Dataset split to extract
        text_keys: Which feature keys contain text. If None, auto-detect.
        output_dir: Where to write the raw JSONL

    Returns:
        Path to the output JSONL file
    """

    # Kill the magic that eats RAM — single sequential read, not iterative training
    read_config = tfds.ReadConfig(
        try_autocaching=False,
        override_buffer_size=1024,
    )

    ds, info = tfds.load(
        dataset_name,
        split=split,
        read_config=read_config,
        with_info=True,
        shuffle_files=False,
    )

    # Auto-detect text features if not specified
    if text_keys is None:
        text_keys = [
            k for k, v in info.features.items() if isinstance(v, (tfds.features.Text,))
        ]
        if not text_keys:
            raise ValueError(
                f"No text features found in {dataset_name}. "
                f"Features: {list(info.features.keys())}. "
                f"Specify text_keys manually."
            )

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{dataset_name.replace('/', '_')}_{split}.jsonl"

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in tfds.as_numpy(ds):
            record = {}
            for key in text_keys:
                val = example[key]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                record[key] = val
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 10_000 == 0:
                print(f"  extracted {count:,} examples...")

    print(f"  {dataset_name}/{split}: {count:,} examples → {out_path}")
    return out_path


def build_corpus(
    jsonl_paths: list[Path],
    output_path: str = "./corpus/train_corpus.txt",
    min_length: int = 64,
    dedup: bool = True,
) -> Path:
    """
    Merge multiple JSONL extractions into a single clean text file.
    One document per line.

    Dedup is content-hash based — exact match only.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    seen_hashes: set[str] = set() if dedup else None
    doc_count = 0
    skip_count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for jsonl_path in jsonl_paths:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    text = " ".join(v for v in record.values() if isinstance(v, str))
                    text = text.strip()

                    if len(text) < min_length:
                        skip_count += 1
                        continue

                    if dedup:
                        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                        if h in seen_hashes:
                            skip_count += 1
                            continue
                        seen_hashes.add(h)

                    out.write(text + "\n")
                    doc_count += 1

    print(
        f"Corpus: {doc_count:,} docs, {skip_count:,} filtered/deduped → {output_path}"
    )
    return Path(output_path)


# =============================================================================
# DATASET REGISTRY — add entries here to pull new datasets
# =============================================================================

DATASETS = [
    # (dataset_name, split, text_keys)
    ("scientific_papers", "train", ["article", "abstract"]),
    ("wiki40b/en", "train", ["text"]),
    ("ag_news_subset", "train", ["description"]),
]


if __name__ == "__main__":
    jsonl_paths = []
    for ds_name, split, keys in DATASETS:
        path = extract_text_from_tfds(ds_name, split=split, text_keys=keys)
        jsonl_paths.append(path)

    build_corpus(jsonl_paths, output_path="./corpus/train_corpus.txt")
