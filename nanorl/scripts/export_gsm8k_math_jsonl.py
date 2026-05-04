#!/usr/bin/env python3
"""Download openai/gsm8k and write nanorl JSONL (RLExample schema).

Prompt format matches the DAPO-Math convention: the model is asked to reason
step by step and place its final answer in \\boxed{}, which the existing
verify_math verifier in data.py already handles.

Ground truth is the integer string that appears after "####" in the GSM8K
answer field, e.g.:
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
    → ground_truth = "72"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Mapping

from datasets import load_dataset
from loguru import logger

DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_CONFIG = "main"

_BOXED_HINT = (
    "\n\nPlease reason step by step, and put your final answer in "
    r"\boxed{}."
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export openai/gsm8k to nanorl JSONL.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/gsm8k.jsonl",
        help="Output .jsonl path (one JSON object per line).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HF dataset id (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Dataset config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name (default: train).",
    )
    return parser


def _parse_answer_field(answer_field: str) -> tuple[str | None, str]:
    """Split a GSM8K answer field into (ground_truth, reasoning).

    Returns (None, reasoning) if '####' is missing or the numeric part is
    unparseable. reasoning is everything before '####', stripped.
    """
    parts = answer_field.split("####")
    reasoning = parts[0].strip()
    if len(parts) < 2:
        return None, reasoning
    raw = parts[-1].strip()
    # Remove thousands-separator commas: "1,234" → "1234"
    raw = raw.replace(",", "")
    # Keep only digits and optional leading minus for negative answers
    m = re.match(r"-?\d+", raw)
    ground_truth = m.group(0) if m else None
    return ground_truth, reasoning


def _build_prompt(question: str) -> str:
    text = question.strip()
    if r"\boxed" not in text:
        text = text + _BOXED_HINT
    return text


def row_to_jsonl_object(
    row: Mapping[str, Any],
    row_index: int,
) -> dict[str, Any] | None:
    question = (row.get("question") or "").strip()
    answer_field = (row.get("answer") or "").strip()

    if not question or not answer_field:
        logger.warning(f"Skipping row {row_index}: empty question or answer field")
        return None

    ground_truth, reasoning = _parse_answer_field(answer_field)
    if ground_truth is None:
        logger.warning(
            f"Skipping row {row_index}: could not find '####' in answer: "
            f"{answer_field[:80]!r}"
        )
        return None

    return {
        "id": f"gsm8k/{row_index}",
        "prompt": _build_prompt(question),
        "ground_truth": ground_truth,
        "meta": {
            "dataset": DEFAULT_DATASET,
            "config": DEFAULT_CONFIG,
            "reasoning": reasoning,
        },
    }


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_and_convert(
    dataset_id: str,
    config_name: str,
    split: str,
) -> list[dict[str, Any]]:
    logger.info(f"Loading {dataset_id=}, {config_name=}, {split=}")
    ds = load_dataset(dataset_id, config_name, split=split)
    out: list[dict[str, Any]] = []
    loaded_rows = 0
    for i, row in enumerate(ds):
        loaded_rows += 1
        obj = row_to_jsonl_object(row, i)
        if obj is not None:
            out.append(obj)
    written_rows = len(out)
    skipped = loaded_rows - written_rows
    logger.info(f"{loaded_rows=}, {written_rows=}, {skipped=}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = load_and_convert(args.dataset, args.config, args.split)
    write_jsonl(args.output, records)
    num_lines = len(records)
    logger.info(f"output={args.output}, {num_lines=}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
