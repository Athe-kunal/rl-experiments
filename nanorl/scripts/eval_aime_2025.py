#!/usr/bin/env python3
"""Evaluate a checkpoint on MathArena/aime_2025 with vLLM and log to W&B."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from statistics import mean
from typing import Any, NamedTuple

import wandb
from datasets import load_dataset
from loguru import logger
from vllm import LLM, SamplingParams

from nanorl.data import extract_last_boxed

_DEFAULT_SYSTEM_PROMPT = (
    "You are a careful competition math solver. "
    "Think step by step before answering. "
    "Then provide the final answer inside \\boxed{...}."
)


class _DatasetColumns(NamedTuple):
    problem_column: str
    answer_column: str


class _EvalStats(NamedTuple):
    accuracy: float
    avg_output_length: float
    total_samples: int
    num_correct: int


@dataclass
class EvalConfig:
    checkpoint_path: str
    num_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    dataset_name: str
    dataset_split: str
    wandb_project: str
    wandb_entity: str | None
    wandb_run_name: str
    seed: int
    dtype: str
    trust_remote_code: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per problem.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dataset-name", type=str, default="MathArena/aime_2025")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--wandb-project", type=str, default="aime-2025-eval")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _load_config(args: argparse.Namespace) -> EvalConfig:
    raw: dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            raw = json.load(f)
    checkpoint_path = args.checkpoint_path or raw.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be set via --checkpoint-path or config JSON")
    num_samples = int(raw.get("num_samples", args.num_samples))
    wandb_run_name = args.wandb_run_name or raw.get("wandb_run_name") or checkpoint_path.split("/")[-1]
    return EvalConfig(
        checkpoint_path=checkpoint_path,
        num_samples=num_samples,
        max_new_tokens=int(raw.get("max_new_tokens", args.max_new_tokens)),
        temperature=float(raw.get("temperature", args.temperature)),
        top_p=float(raw.get("top_p", args.top_p)),
        dataset_name=str(raw.get("dataset_name", args.dataset_name)),
        dataset_split=str(raw.get("dataset_split", args.dataset_split)),
        wandb_project=str(raw.get("wandb_project", args.wandb_project)),
        wandb_entity=raw.get("wandb_entity", args.wandb_entity),
        wandb_run_name=wandb_run_name,
        seed=int(raw.get("seed", args.seed)),
        dtype=str(raw.get("dtype", args.dtype)),
        trust_remote_code=bool(raw.get("trust_remote_code", args.trust_remote_code)),
    )


def _detect_columns(row: dict[str, Any]) -> _DatasetColumns:
    problem_candidates = ["problem", "question", "prompt"]
    answer_candidates = ["answer", "final_answer", "solution"]
    problem_column = next((name for name in problem_candidates if name in row), "")
    answer_column = next((name for name in answer_candidates if name in row), "")
    if not problem_column or not answer_column:
        raise ValueError(f"Unsupported dataset schema, keys={list(row.keys())}")
    return _DatasetColumns(problem_column=problem_column, answer_column=answer_column)


def _build_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def _evaluate_predictions(ground_truth: str, responses: list[str]) -> tuple[int, float]:
    gt = " ".join(ground_truth.strip().split())
    correct_count = 0
    avg_length = mean(len(response) for response in responses) if responses else 0.0
    for response in responses:
        pred = extract_last_boxed(response)
        pred_norm = " ".join((pred or "").strip().split())
        if pred is not None and pred_norm == gt:
            correct_count += 1
    return correct_count, avg_length


def run_evaluation(config: EvalConfig) -> _EvalStats:
    logger.info(f"{config=}")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    first_row = dataset[0]
    columns = _detect_columns(first_row)
    logger.info(f"{columns=}")
    llm = LLM(
        model=config.checkpoint_path,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        seed=config.seed,
    )
    sampling_params = SamplingParams(
        n=config.num_samples,
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    total_correct = 0
    output_lengths: list[float] = []
    total_samples = 0

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config=config.__dict__,
    )

    for row_index, row in enumerate(dataset):
        problem = str(row[columns.problem_column])
        ground_truth = str(row[columns.answer_column])
        outputs = llm.chat(messages=_build_messages(problem), sampling_params=sampling_params)
        responses = [candidate.text for candidate in outputs[0].outputs]
        correct_count, avg_length = _evaluate_predictions(ground_truth, responses)
        output_lengths.append(avg_length)
        total_correct += correct_count
        total_samples += len(responses)

        row_accuracy = float(correct_count) / float(len(responses)) if responses else 0.0
        wandb.log({
            "eval/row_index": row_index,
            "eval/row_accuracy": row_accuracy,
            "eval/row_avg_output_length": avg_length,
            "eval/total_correct": total_correct,
            "eval/total_samples": total_samples,
        })

    run.finish()
    accuracy = float(total_correct) / float(total_samples) if total_samples else 0.0
    avg_output_length = mean(output_lengths) if output_lengths else 0.0
    return _EvalStats(
        accuracy=accuracy,
        avg_output_length=avg_output_length,
        total_samples=total_samples,
        num_correct=total_correct,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    config = _load_config(args)
    stats = run_evaluation(config)
    logger.info(f"{stats=}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
