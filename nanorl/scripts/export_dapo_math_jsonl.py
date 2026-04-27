#!/usr/bin/env python3
"""Download open-r1/DAPO-Math-17k-Processed and write nanorl JSONL (RLExample schema)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Mapping

from datasets import load_dataset

DEFAULT_DATASET = "open-r1/DAPO-Math-17k-Processed"
CONFIG_CHOICES = ("all", "en", "cn")

# HF `prompt` is often problem-only; nanorl rewards parse \boxed{...} in completions.
_BOXED_HINT = (
    "\n\nPlease reason step by step, and put your final answer in "
    r"\boxed{}."
)

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Hugging Face DAPO-Math-17k-Processed to nanorl JSONL.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output .jsonl path (one JSON object per line).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HF dataset id (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--config",
        default="all",
        choices=CONFIG_CHOICES,
        help="Dataset config: all, en, or cn.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name (default: train).",
    )
    parser.add_argument(
        "--no-append-boxed-hint",
        action="store_true",
        help="Do not append the \\boxed{} instruction when the prompt lacks it.",
    )
    return parser


def _stable_example_id(row: Mapping[str, Any], row_index: int) -> str:
    top_extra = row.get("extra_info") or {}
    reward_model = row.get("reward_model") or {}
    nested_extra = reward_model.get("extra_info") or {}
    raw = top_extra.get("index") or nested_extra.get("index")
    if raw is not None and str(raw).strip():
        return f"dapo_math/{raw}"
    return f"dapo_math/row_{row_index}"


def _ground_truth_string(row: Mapping[str, Any]) -> str:
    reward_model = row.get("reward_model") or {}
    sol = (row.get("solution") or "").strip()
    gt = (reward_model.get("ground_truth") or "").strip()
    return sol or gt


def _prompt_for_nanorl(raw_prompt: str, append_boxed_hint: bool) -> str:
    text = raw_prompt.strip()
    if append_boxed_hint and r"\boxed" not in text:
        text = text + _BOXED_HINT
    return text


def row_to_jsonl_object(
    row: Mapping[str, Any],
    row_index: int,
    config_name: str,
    append_boxed_hint: bool,
) -> dict[str, Any] | None:
    prompt = _prompt_for_nanorl(row.get("prompt") or "", append_boxed_hint)
    ground_truth = _ground_truth_string(row)
    if not prompt or not ground_truth:
        log.warning(
            "skipping row with empty prompt or ground_truth: %s",
            {"row_index": row_index, "id_hint": _stable_example_id(row, row_index)},
        )
        return None
    meta: dict[str, Any] = {"hf_config": config_name}
    for key in ("data_source", "ability"):
        val = row.get(key)
        if val is not None:
            meta[key] = val
    return {
        "id": _stable_example_id(row, row_index),
        "prompt": prompt,
        "ground_truth": ground_truth,
        "meta": meta,
    }


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_and_convert(
    dataset_id: str,
    config_name: str,
    split: str,
    append_boxed_hint: bool,
) -> list[dict[str, Any]]:
    log.info(
        "%s",
        {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split": split,
        },
    )
    ds = load_dataset(dataset_id, config_name, split=split)
    out: list[dict[str, Any]] = []
    loaded_rows = 0
    for i, row in enumerate(ds):
        loaded_rows += 1
        obj = row_to_jsonl_object(row, i, config_name, append_boxed_hint)
        if obj is not None:
            out.append(obj)
    log.info("%s", {"loaded_rows": loaded_rows, "written_rows": len(out)})
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    records = load_and_convert(
        args.dataset,
        args.config,
        args.split,
        append_boxed_hint=not args.no_append_boxed_hint,
    )
    write_jsonl(args.output, records)
    log.info("%s", {"output": args.output, "num_lines": len(records)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
