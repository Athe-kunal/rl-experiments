"""Prepare DAPO math training data from open-r1/DAPO-Math-17k-Processed.

Usage:
  uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.prepare_openr1_dapo_math_data \
    --data-dir ./data/dapo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="data/dapo/")
    parser.add_argument("--train-output", type=str, default="dapo-math-17k-openr1-processed.parquet")
    parser.add_argument("--val-output", type=str, default="aime-2024.parquet")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / args.train_output
    val_path = data_dir / args.val_output

    ds = load_dataset("open-r1/DAPO-Math-17k-Processed")
    # HF dataset usually exposes a single "train" split for this corpus.
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    # Dedupe on identity columns only: open-r1 rows include ndarray (e.g. source_prompt)
    # and dict columns that break full-frame drop_duplicates (unhashable).
    before = len(df)
    df["_dedupe_rm"] = df["reward_model"].apply(
        lambda x: json.dumps(x, sort_keys=True, default=str) if isinstance(x, dict) else repr(x)
    )
    df = df.drop_duplicates(subset=["data_source", "prompt", "ability", "_dedupe_rm"]).drop(
        columns=["_dedupe_rm"]
    )
    after = len(df)
    df.to_parquet(train_path, index=False)

    # Keep evaluation set consistent with existing DAPO scripts.
    aime_df = pd.read_parquet(
        "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
    )
    aime_df.to_parquet(val_path, index=False)

    print(f"Wrote train file: {train_path} rows={after} (deduped {before - after})")
    print(f"Wrote val file:   {val_path} rows={len(aime_df)}")


if __name__ == "__main__":
    main()
