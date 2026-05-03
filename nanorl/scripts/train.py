"""
Minimal RL training loop for HuggingFace causal-LM models.

Rollout generation is delegated to a separate vLLM worker process
(`nanorl/scripts/rollout_worker.py`) on its own GPU. The trainer handles
model loading (HF), the optimization loop, and checkpoint saving. Use the
launcher at `nanorl/runs/train.sh` to start both processes.

Everything tweakable lives under nanorl/:
  - nanorl/loss.py    : pluggable loss functions (GRPO, DAPO, REINFORCE)
  - nanorl/rollout.py : remote-rollout helpers, log-prob scoring, batch packing
  - nanorl/data.py    : RL dataset + reward interface
"""

import os
import math
import time
import asyncio
import argparse
import itertools
from typing import NamedTuple
from statistics import mean

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import autodetect_device_type, compute_init, compute_cleanup, print0
from nanorl.loss import ALGORITHMS, compute_advantages
from nanorl.rollout import (
    get_logprobs,
    generate_rollouts_remote,
    generate_rollouts_remote_async,
    remote_vllm_init_weight_transfer,
    prepare_batch,
    sync_weights_to_vllm_inplace,
    wait_for_rollout_worker,
)
from nanorl.data import (
    build_rl_dataset,
    distributed_rl_loader,
    RewardWorkerPool,
    apply_overlong_shaping,
    split_rl_dataset,
)
from nanorl.weave_logging import WeaveTrajectoryLogger
from nanorl.wandb_logging import WandbTrainingLogger


class _BroadcastBatchResult(NamedTuple):
    batch: dict[str, torch.Tensor]
    advantages: torch.Tensor
    mean_reward: float


class _ValidationStats(NamedTuple):
    mean_reward: float
    response_len_mean: float
    truncation_ratio: float
    duration_s: float


class _WeightTransferInitConfig(NamedTuple):
    master_addr: str
    weight_transfer_port: int
    world_size: int


def _compute_effective_eval_prompts(
    *,
    eval_prompts_per_step: int,
    prompts_per_step: int,
    validation_dataset_size: int,
) -> int:
    """Pick a safe validation batch size for the distributed loader."""
    requested = eval_prompts_per_step if eval_prompts_per_step > 0 else prompts_per_step
    if validation_dataset_size <= 0:
        return 0
    return min(requested, validation_dataset_size)


def _run_validation_step(
    *,
    args: argparse.Namespace,
    loader,
    rewarder: RewardWorkerPool,
    weave_logger: WeaveTrajectoryLogger,
    step: int,
) -> _ValidationStats:
    """Run one validation rollout/reward pass on rank 0."""
    validation_t0 = time.time()
    examples, _loader_state = next(loader)
    prompt_texts = [example.prompt for example in examples]
    rollouts = asyncio.run(generate_rollouts_remote_async(
        base_url=args.rollout_worker_url,
        prompts=prompt_texts,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        prompt_chunk_size=args.eval_async_prompt_chunk_size,
        max_in_flight_requests=args.eval_async_max_in_flight,
    ))
    responses = [rollout["response"] for rollout in rollouts]
    expanded_examples = [examples[i // args.num_samples] for i in range(len(rollouts))]
    rewards, infos = rewarder.score(expanded_examples, responses, step=step)
    response_lengths = [len(rollout["response_ids"]) for rollout in rollouts]
    shaped_rewards = apply_overlong_shaping(rewards, response_lengths, args.max_new_tokens)
    weave_logger.log_trajectories(
        step=step,
        run_name=f"{args.run_name}_validation",
        examples=expanded_examples,
        responses=responses,
        reward_infos=infos,
        verifier_rewards=rewards,
        training_rewards=shaped_rewards,
        num_samples_per_prompt=args.num_samples,
        rollouts=rollouts,
        max_new_tokens=args.max_new_tokens,
    )
    mean_reward = sum(shaped_rewards) / len(shaped_rewards) if shaped_rewards else 0.0
    response_len_mean = mean(response_lengths) if response_lengths else 0.0
    num_truncated = sum(1 for length in response_lengths if length >= args.max_new_tokens)
    truncation_ratio = (num_truncated / len(response_lengths)) if response_lengths else 0.0
    duration_s = time.time() - validation_t0
    return _ValidationStats(
        mean_reward=mean_reward,
        response_len_mean=response_len_mean,
        truncation_ratio=truncation_ratio,
        duration_s=duration_s,
    )


def summarize_rewards(rewards: list[float]) -> str:
    if not rewards:
        return "n=0"
    nonzero = sum(1 for r in rewards if r != 0.0)
    return (
        f"n={len(rewards)} "
        f"mean={mean(rewards):.4f} "
        f"min={min(rewards):.4f} "
        f"max={max(rewards):.4f} "
        f"nonzero={nonzero}/{len(rewards)} ({nonzero / len(rewards):.1%})"
    )


def broadcast_batch(
    master_process: bool,
    ddp: bool,
    device: torch.device,
    ddp_rank: int,
    ddp_world_size: int,
    batch: dict[str, torch.Tensor] | None,
    advantages: torch.Tensor | None,
    mean_reward: float,
) -> _BroadcastBatchResult:
    """Broadcast a rollout batch prepared by rank 0 to every training rank."""
    del ddp_rank  # unused, kept in signature for caller clarity.
    del ddp_world_size  # unused, kept in signature for caller clarity.
    if master_process:
        assert batch is not None
        assert advantages is not None
        batch = {k: v.to(device) for k, v in batch.items()}
        advantages = advantages.to(device)
        meta = torch.tensor(
            [batch["input_ids"].shape[0], batch["input_ids"].shape[1]],
            dtype=torch.long,
            device=device,
        )
        reward_meta = torch.tensor([mean_reward], dtype=torch.float32, device=device)
    else:
        meta = torch.zeros(2, dtype=torch.long, device=device)
        reward_meta = torch.zeros(1, dtype=torch.float32, device=device)

    if ddp:
        dist.broadcast(meta, src=0)
        dist.broadcast(reward_meta, src=0)

    total_samples, max_len = meta.tolist()
    if not master_process:
        batch = {
            "input_ids": torch.empty((total_samples, max_len), dtype=torch.long, device=device),
            "attention_mask": torch.empty((total_samples, max_len), dtype=torch.long, device=device),
            "response_mask": torch.empty((total_samples, max_len), dtype=torch.float32, device=device),
            "rewards": torch.empty(total_samples, dtype=torch.float32, device=device),
        }
        advantages = torch.empty(total_samples, dtype=torch.float32, device=device)

    if ddp:
        dist.broadcast(batch["input_ids"], src=0)
        dist.broadcast(batch["attention_mask"], src=0)
        dist.broadcast(batch["response_mask"], src=0)
        dist.broadcast(batch["rewards"], src=0)
        assert advantages is not None
        dist.broadcast(advantages, src=0)

    assert batch is not None
    assert advantages is not None
    return _BroadcastBatchResult(
        batch=batch,
        advantages=advantages,
        mean_reward=float(reward_meta.item()),
    )


def local_shard(
    batch: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    ddp_rank: int,
    ddp_world_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    total_samples = batch["input_ids"].shape[0]
    per_rank = total_samples // ddp_world_size
    start = ddp_rank * per_rank
    end = start + per_rank
    local_batch = {k: v[start:end] for k, v in batch.items()}
    local_advantages = advantages[start:end]
    return local_batch, local_advantages


if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(description="RL training for HF models")
    # Model
    parser.add_argument("--model", type=str, required=True, help="HF model path (e.g. Qwen/Qwen3-0.6B)")
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="grpo", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--clip", type=float, default=0.2, help="PPO/GRPO clip range")
    parser.add_argument("--kl-coeff", type=float, default=0.0, help="KL penalty coefficient")
    # Generation
    parser.add_argument("--num-samples", type=int, default=16, help="Completions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rollout-worker-url", type=str, default="http://127.0.0.1:8047")
    parser.add_argument("--rollout-worker-world-size", type=int, default=1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of RL steps (-1 to exhaust the full loader)")
    parser.add_argument("--prompts-per-step", type=int, default=8, help="Prompts per RL step")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training micro-batch size")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="Optimizer steps per rollout batch")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-prompts-per-step", type=int, default=0)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--eval-async-prompt-chunk-size", type=int, default=64)
    parser.add_argument("--eval-async-max-in-flight", type=int, default=8)
    # Task / Reward
    parser.add_argument("--reward-workers", type=int, default=0, help="Reward worker pool size")
    # Runtime
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run-name", type=str, default="dummy", help="wandb run name")
    parser.add_argument("--weave-project", type=str, default="", help="W&B Weave project name")
    parser.add_argument("--wandb-project", type=str, default="", help="W&B project name for scalar metrics")
    parser.add_argument("--save-dir", type=str, default="rl_checkpoints")
    parser.add_argument("--save-every", type=int, default=0, help="Save a checkpoint every N steps (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Compute init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    print0(f"Loading model: {args.model}")
    print0(f"Algorithm: {args.algorithm}")
    print0(f"Device: {device}, World size: {ddp_world_size}")

    global_rollout_samples = args.prompts_per_step * args.num_samples
    assert global_rollout_samples % ddp_world_size == 0, (
        f"prompts_per_step * num_samples ({global_rollout_samples}) must be divisible by "
        f"world_size ({ddp_world_size})"
    )

    # -----------------------------------------------------------------------------
    # Tokenizer + HF training model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # -----------------------------------------------------------------------------
    # Remote vLLM rollout worker + NCCL weight transfer setup.
    model_update_group = None
    weight_transfer_init_config = _WeightTransferInitConfig(
        master_addr="",
        weight_transfer_port=0,
        world_size=0,
    )
    trainer_world_size = ddp_world_size
    world_size = trainer_world_size + args.rollout_worker_world_size

    if master_process:
        health = wait_for_rollout_worker(args.rollout_worker_url, timeout_s=300)
        print0(f"Remote vLLM rollout worker ready: {health['model_path']}")
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        # Pick a free port for the weight-transfer rendezvous, separate from
        # MASTER_PORT which torchrun already has bound.
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as socket_server:
            socket_server.bind((master_addr, 0))
            weight_transfer_port = socket_server.getsockname()[1]
        weight_transfer_init_config = _WeightTransferInitConfig(
            master_addr=master_addr,
            weight_transfer_port=weight_transfer_port,
            world_size=world_size,
        )

    if ddp:
        broadcast_values = [weight_transfer_init_config]
        dist.broadcast_object_list(broadcast_values, src=0)
        weight_transfer_init_config = broadcast_values[0]

    if master_process:
        remote_vllm_init_weight_transfer(
            args.rollout_worker_url,
            master_address=weight_transfer_init_config.master_addr,
            master_port=weight_transfer_init_config.weight_transfer_port,
            rank_offset=trainer_world_size,
            world_size=weight_transfer_init_config.world_size,
        )

    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    model_update_group = NCCLWeightTransferEngine.trainer_init({
        "master_address": weight_transfer_init_config.master_addr,
        "master_port": weight_transfer_init_config.weight_transfer_port,
        "world_size": weight_transfer_init_config.world_size,
    })
    print0("NCCL weight transfer group initialized.")

    # Ensure step-0 rollouts use model weights instead of dummy/random init.
    sync_weights_to_vllm_inplace(
        train_model=raw_model,
        base_url=args.rollout_worker_url,
        model_update_group=model_update_group,
        packed=True,
        fsdp=False,
        remote_update_enabled=master_process,
    )
    print0("Initial trainer weights synced to rollout worker.")

    # -----------------------------------------------------------------------------
    # RL dataset + reward worker pool + loss fn
    if master_process:
        dataset = build_rl_dataset()
        dataset_split = split_rl_dataset(
            dataset,
            validation_size=args.validation_size,
            seed=args.seed,
        )
        train_dataset = dataset_split.train_dataset
        validation_dataset = dataset_split.validation_dataset
        loader = distributed_rl_loader(
            train_dataset,
            prompts_per_step=args.prompts_per_step,
            world_size=1,
            rank=0,
            seed=args.seed,
        )
        validation_prompts_per_step = _compute_effective_eval_prompts(
            eval_prompts_per_step=args.eval_prompts_per_step,
            prompts_per_step=args.prompts_per_step,
            validation_dataset_size=len(validation_dataset),
        )
        validation_loader = None
        if validation_prompts_per_step > 0:
            validation_loader = distributed_rl_loader(
                validation_dataset,
                prompts_per_step=validation_prompts_per_step,
                world_size=1,
                rank=0,
                seed=args.seed + 17,
            )
        rewarder = RewardWorkerPool(num_workers=args.reward_workers)
        print0(f"{len(train_dataset)=}")
        print0(f"{len(validation_dataset)=}")
        print0(f"{validation_prompts_per_step=}")
        print0(f"{args.eval_async_prompt_chunk_size=}")
        print0(f"{args.eval_async_max_in_flight=}")
    else:
        loader = None
        validation_loader = None
        rewarder = None
    loss_fn = ALGORITHMS[args.algorithm]

    # -----------------------------------------------------------------------------
    # Optimizer
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )

    # Linear warmup over 20 rollout steps (DAPO paper §4.1).
    _warmup_steps = 20
    def _lr_lambda(current_step: int) -> float:
        if current_step < _warmup_steps:
            return float(current_step + 1) / float(_warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # -----------------------------------------------------------------------------
    # Training loop
    print0(f"Starting RL training: {args.num_steps} steps, {args.prompts_per_step} prompts/step, {args.num_samples} samples/prompt")
    weave_logger = WeaveTrajectoryLogger(project_name=args.weave_project, enabled=master_process)
    wandb_project = args.wandb_project or args.weave_project
    wandb_logger = WandbTrainingLogger(
        project_name=wandb_project,
        run_name=args.run_name,
        config=vars(args),
        enabled=master_process,
    )
    print0(f"{args.weave_project=}")
    print0(f"{wandb_project=}")
    print0(f"{weave_logger.is_enabled=}")
    print0(f"{wandb_logger.is_enabled=}")

    train_start_time = time.time()
    step_iter = itertools.count() if args.num_steps == -1 else range(args.num_steps)
    for step in step_iter:
        t0 = time.time()
        phase = {}
        _ds_rounds = 0
        num_valid_groups = 0
        _loader_state = None
        skip_update_master = False

        if master_process:
            # === Dynamic Sampling (DAPO §3.2) ===
            # Re-sample batches until `prompts_per_step` groups with mixed accuracy
            # are collected. Groups where all samples are correct (accuracy=1) or all
            # wrong (accuracy=0) produce zero advantage and are discarded.
            phase_t0 = time.time()
            ds_examples: list = []
            ds_rollouts: list[dict] = []
            ds_rewards_raw: list[float] = []
            ds_infos: list[dict] = []

            while len(ds_examples) < args.prompts_per_step:
                _ds_rounds += 1
                if _ds_rounds > 20:
                    print0(
                        f"[step {step:04d}] dynamic sampling: gave up after 20 rounds "
                        f"({len(ds_examples)}/{args.prompts_per_step} valid groups collected)"
                    )
                    break

                round_examples, _loader_state = next(loader)
                round_prompts = [ex.prompt for ex in round_examples]
                round_rollouts = generate_rollouts_remote(
                    args.rollout_worker_url, round_prompts,
                    args.num_samples, args.max_new_tokens,
                    args.temperature, args.top_k,
                )
                round_expanded = [
                    round_examples[i // args.num_samples]
                    for i in range(len(round_rollouts))
                ]
                round_responses = [r["response"] for r in round_rollouts]
                round_rewards_raw, round_infos = rewarder.score(
                    round_expanded, round_responses, step=step
                )

                for gi in range(len(round_examples)):
                    if len(ds_examples) >= args.prompts_per_step:
                        break
                    g_start = gi * args.num_samples
                    g_end = g_start + args.num_samples
                    g_rewards = round_rewards_raw[g_start:g_end]
                    # Reward is +1 (correct) or -1 (wrong); keep groups with mixed outcomes.
                    num_correct = sum(1 for r in g_rewards if r > 0)
                    if 0 < num_correct < args.num_samples:
                        ds_examples.append(round_examples[gi])
                        ds_rollouts.extend(round_rollouts[g_start:g_end])
                        ds_rewards_raw.extend(g_rewards)
                        ds_infos.extend(round_infos[g_start:g_end])

            phase["dynamic_sampling_s"] = time.time() - phase_t0
            num_valid_groups = len(ds_examples)
            skip_update_master = num_valid_groups == 0

            if not skip_update_master:
                examples = ds_examples
                rollouts = ds_rollouts
                resp_lens = [len(r["response_ids"]) for r in rollouts]
                n_truncated = sum(1 for n in resp_lens if n >= args.max_new_tokens)

                phase_t0 = time.time()
                expanded_examples = [
                    examples[i // args.num_samples] for i in range(len(rollouts))
                ]
                responses = [r["response"] for r in rollouts]
                shaped_rewards = apply_overlong_shaping(ds_rewards_raw, resp_lens, args.max_new_tokens)
                weave_logger.log_trajectories(
                    step=step,
                    run_name=args.run_name,
                    examples=expanded_examples,
                    responses=responses,
                    reward_infos=ds_infos,
                    verifier_rewards=ds_rewards_raw,
                    training_rewards=shaped_rewards,
                    num_samples_per_prompt=args.num_samples,
                    rollouts=rollouts,
                    max_new_tokens=args.max_new_tokens,
                )
                rewards = shaped_rewards
                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                phase["reward_shaping_s"] = time.time() - phase_t0
                reward_summary = summarize_rewards(rewards)

                phase_t0 = time.time()
                batch = prepare_batch(rollouts, rewards, tokenizer, args.max_seq_len, "cpu")

                # Overlong Filtering (DAPO §3.4): zero the response_mask for sequences
                # that hit the generation cap so truncated reasoning doesn't contribute
                # to the policy gradient. Soft reward punishment was already applied above.
                for bi, rlen in enumerate(resp_lens):
                    if rlen >= args.max_new_tokens:
                        batch["response_mask"][bi] = 0.0

                advantages = compute_advantages(
                    args.algorithm,
                    batch["rewards"],
                    num_samples_per_prompt=args.num_samples,
                )
                phase["pack_batch_s"] = time.time() - phase_t0

                print0(
                    f"[step {step:04d}] ds_rounds={_ds_rounds} "
                    f"valid_groups={num_valid_groups} "
                    f"truncated={n_truncated}/{len(resp_lens)} "
                    f"rewards[{reward_summary}]"
                )
            else:
                print0(
                    f"[step {step:04d}] dynamic sampling: no valid groups after "
                    f"{_ds_rounds} rounds — skipping gradient update"
                )
                examples, rollouts, resp_lens, n_truncated = [], [], [], 0
                rewards, mean_reward, reward_summary = [], 0.0, "n=0"
                batch, advantages = None, None
        else:
            examples, rollouts, resp_lens, n_truncated = [], [], [], 0
            rewards, mean_reward, reward_summary = [], 0.0, ""
            batch, advantages = None, None

        # Broadcast skip flag so all DDP ranks agree before the weight-transfer barrier.
        skip_update = skip_update_master
        if ddp:
            skip_tensor = torch.tensor(
                [1 if skip_update_master else 0], dtype=torch.long, device=device
            )
            dist.broadcast(skip_tensor, src=0)
            skip_update = bool(skip_tensor.item())

        if not skip_update:
            phase_t0 = time.time()
            broadcast_result = broadcast_batch(
                master_process, ddp, device, ddp_rank, ddp_world_size,
                batch, advantages, mean_reward,
            )
            batch = broadcast_result.batch
            advantages = broadcast_result.advantages
            mean_reward = broadcast_result.mean_reward
            batch, advantages = local_shard(batch, advantages, ddp_rank, ddp_world_size)
            local_bsz = batch["input_ids"].shape[0]
            phase["broadcast_and_shard_s"] = time.time() - phase_t0

            # Old log-probs: with ppo_epochs == 1 ratio is always 1 and clipping
            # never fires; with >1 epochs we freeze a snapshot so Clip-Higher and
            # Clip-Low actually activate on subsequent passes over the same rollout.
            total_samples = batch["input_ids"].shape[0]
            micro_bs = args.train_batch_size
            n_microbatches = math.ceil(total_samples / micro_bs)

            phase_t0 = time.time()
            if args.ppo_epochs > 1:
                old_logprobs_chunks = []
                with torch.no_grad():
                    for mb in range(n_microbatches):
                        start = mb * micro_bs
                        end = min(start + micro_bs, total_samples)
                        lp_mb, _ = get_logprobs(
                            raw_model,
                            batch["input_ids"][start:end],
                            batch["attention_mask"][start:end],
                            batch["response_mask"][start:end],
                        )
                        old_logprobs_chunks.append(lp_mb)
                old_logprobs = torch.cat(old_logprobs_chunks, dim=0)
            else:
                old_logprobs = None
            phase["old_logprobs_s"] = time.time() - phase_t0

            phase_t0 = time.time()
            total_loss = 0.0
            grad_norms: list[float] = []
            for _epoch in range(args.ppo_epochs):
                optimizer.zero_grad()
                for mb in range(n_microbatches):
                    start = mb * micro_bs
                    end = min(start + micro_bs, total_samples)
                    mb_ids = batch["input_ids"][start:end]
                    mb_attn = batch["attention_mask"][start:end]
                    mb_resp = batch["response_mask"][start:end]
                    mb_advantages = advantages[start:end]

                    logprobs, shift_mask = get_logprobs(model, mb_ids, mb_attn, mb_resp)
                    mb_old_lp = (
                        logprobs.detach() if old_logprobs is None
                        else old_logprobs[start:end]
                    )
                    loss = loss_fn(
                        logprobs=logprobs,
                        old_logprobs=mb_old_lp,
                        advantages=mb_advantages,
                        response_mask=shift_mask,
                        clip=args.clip,
                        kl_coeff=args.kl_coeff,
                    ) / n_microbatches
                    loss.backward()
                    total_loss += loss.item()

                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                grad_norms.append(float(grad_norm))
                optimizer.step()
            phase["update_s"] = time.time() - phase_t0
        else:
            local_bsz = 0
            total_loss = 0.0
            grad_norms = []
            phase.setdefault("broadcast_and_shard_s", 0.0)
            phase.setdefault("old_logprobs_s", 0.0)
            phase.setdefault("update_s", 0.0)

        # Advance LR schedule every step (including skipped ones).
        scheduler.step()

        # Sync updated weights to vLLM for the next rollout step.
        phase_t0 = time.time()
        sync_weights_to_vllm_inplace(
            train_model=raw_model,
            base_url=args.rollout_worker_url,
            model_update_group=model_update_group,
            packed=True,
            fsdp=False,
            remote_update_enabled=master_process,
        )
        phase["sync_rollout_s"] = time.time() - phase_t0

        dt = time.time() - t0
        avg_loss = total_loss / max(args.ppo_epochs, 1)
        grad_norm_mean = mean(grad_norms) if grad_norms else 0.0
        grad_norm_max = max(grad_norms) if grad_norms else 0.0
        response_len_mean = mean(resp_lens) if master_process and resp_lens else 0.0
        truncation_ratio = (
            (n_truncated / len(resp_lens)) if master_process and resp_lens else 0.0
        )
        train_elapsed_s = time.time() - train_start_time

        print0(
            f"step {step:04d}/{'inf' if args.num_steps == -1 else f'{args.num_steps:04d}'} | "
            f"loss: {avg_loss:.4f} | reward: {mean_reward:.4f} "
            f"| local_bsz: {local_bsz} | ds_rounds: {_ds_rounds} | dt: {dt:.1f}s "
            f"| lr: {scheduler.get_last_lr()[0]:.2e}"
        )

        wandb_metrics = {
            "train/loss": avg_loss,
            "train/avg_reward": mean_reward,
            "train/grad_norm_mean": grad_norm_mean,
            "train/grad_norm_max": grad_norm_max,
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/step_time_s": dt,
            "train/elapsed_s": train_elapsed_s,
            "train/local_batch_size": float(local_bsz),
            "train/response_len_mean": response_len_mean,
            "train/truncation_ratio": truncation_ratio,
            "train/dynamic_sampling_rounds": float(_ds_rounds),
            "train/valid_groups": float(num_valid_groups),
            "timing/dynamic_sampling_s": phase.get("dynamic_sampling_s", 0.0),
            "timing/reward_shaping_s": phase.get("reward_shaping_s", 0.0),
            "timing/pack_batch_s": phase.get("pack_batch_s", 0.0),
            "timing/broadcast_and_shard_s": phase.get("broadcast_and_shard_s", 0.0),
            "timing/old_logprobs_s": phase.get("old_logprobs_s", 0.0),
            "timing/update_s": phase.get("update_s", 0.0),
            "timing/sync_rollout_s": phase["sync_rollout_s"],
        }

        if (
            master_process
            and validation_loader is not None
            and args.eval_every > 0
            and (step + 1) % args.eval_every == 0
        ):
            print0("Evaluating on validation split...")
            validation_stats = _run_validation_step(
                args=args,
                loader=validation_loader,
                rewarder=rewarder,
                weave_logger=weave_logger,
                step=step,
            )
            validation_metrics = {
                "validation/mean_reward": validation_stats.mean_reward,
                "validation/response_len_mean": validation_stats.response_len_mean,
                "validation/truncation_ratio": validation_stats.truncation_ratio,
                "validation/eval_time_s": validation_stats.duration_s,
            }
            wandb_metrics.update(validation_metrics)
            print0(f"{validation_stats=}")

        wandb_logger.log_metrics(step=step, metrics=wandb_metrics)

        if args.save_every > 0 and (step + 1) % args.save_every == 0 and master_process:
            step_path = os.path.join(args.save_dir, f"step_{step+1:06d}")
            raw_model.save_pretrained(step_path, safe_serialization=True)
            tokenizer.save_pretrained(step_path)
            print0(f"[step {step:04d}] saved checkpoint to {step_path}")

        if args.num_steps == -1:
            _stop = torch.zeros(1, dtype=torch.long, device=device)
            if master_process and _loader_state is not None:
                _stop[0] = 1 if _loader_state["epoch"] > 0 else 0
            if ddp:
                dist.broadcast(_stop, src=0)
            if _stop.item():
                print0(f"[step {step:04d}] loader exhausted, stopping.")
                break

    # -----------------------------------------------------------------------------
    # Save
    if master_process:
        save_path = os.path.join(args.save_dir, f"{args.model.replace('/', '_')}_{args.algorithm}")
        os.makedirs(save_path, exist_ok=True)
        raw_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print0(f"Saved to {save_path}")

    wandb_logger.finish()
    if rewarder is not None:
        rewarder.close()
    compute_cleanup()
