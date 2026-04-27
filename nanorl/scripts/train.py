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
import argparse
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
    materialize_rollout_checkpoint,
    remote_vllm_reload,
    prepare_batch,
    wait_for_rollout_worker,
)
from nanorl.data import build_rl_dataset, distributed_rl_loader, RewardWorkerPool, apply_overlong_shaping
from nanorl.weave_logging import WeaveTrajectoryLogger
from nanorl.wandb_logging import WandbTrainingLogger


class _BroadcastBatchResult(NamedTuple):
    batch: dict[str, torch.Tensor]
    advantages: torch.Tensor
    mean_reward: float


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
    parser.add_argument("--rollout-sync-dir", type=str, default="")
    # Training
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of RL steps")
    parser.add_argument("--prompts-per-step", type=int, default=8, help="Prompts per RL step")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training micro-batch size")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="Optimizer steps per rollout batch")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=20)
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

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # -----------------------------------------------------------------------------
    # Remote vLLM rollout worker (rank 0 handshake)
    if master_process:
        health = wait_for_rollout_worker(args.rollout_worker_url, timeout_s=300)
        print0(f"Remote vLLM rollout worker ready: {health['model_path']}")

    # -----------------------------------------------------------------------------
    # RL dataset + reward worker pool + loss fn
    if master_process:
        dataset = build_rl_dataset()
        loader = distributed_rl_loader(
            dataset,
            prompts_per_step=args.prompts_per_step,
            world_size=1,
            rank=0,
            seed=args.seed,
        )
        rewarder = RewardWorkerPool(num_workers=args.reward_workers)
    else:
        loader = None
        rewarder = None
    loss_fn = ALGORITHMS[args.algorithm]

    # -----------------------------------------------------------------------------
    # Optimizer
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )


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
    for step in range(args.num_steps):
        t0 = time.time()
        phase = {}

        if master_process:
            # 1. Sample prompts for the whole global batch on rank 0
            phase_t0 = time.time()
            examples, _loader_state = next(loader)
            prompt_texts = [ex.prompt for ex in examples]
            phase["fetch_prompts_s"] = time.time() - phase_t0

            # 2. Generate rollouts via the remote vLLM worker
            phase_t0 = time.time()
            rollouts = generate_rollouts_remote(
                args.rollout_worker_url, prompt_texts,
                args.num_samples, args.max_new_tokens,
                args.temperature, args.top_k,
            )
            phase["rollout_s"] = time.time() - phase_t0
            resp_lens = [len(r["response_ids"]) for r in rollouts]
            n_truncated = sum(1 for n in resp_lens if n >= args.max_new_tokens)
            print0(f"[step {step:04d}] truncated={n_truncated}/{len(resp_lens)}")

            # 3. Compute rewards on rank 0
            phase_t0 = time.time()
            expanded_examples = [examples[i // args.num_samples] for i in range(len(rollouts))]
            responses = [r["response"] for r in rollouts]
            rewards, _infos = rewarder.score(expanded_examples, responses, step=step)
            shaped_rewards = apply_overlong_shaping(rewards, resp_lens, args.max_new_tokens)
            weave_logger.log_trajectories(
                step=step,
                run_name=args.run_name,
                examples=expanded_examples,
                responses=responses,
                reward_infos=_infos,
                verifier_rewards=rewards,
                training_rewards=shaped_rewards,
                num_samples_per_prompt=args.num_samples,
                rollouts=rollouts,
                max_new_tokens=args.max_new_tokens,
            )
            rewards = shaped_rewards
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            phase["reward_s"] = time.time() - phase_t0
            reward_summary = summarize_rewards(rewards)

            # 4. Pack training batch on CPU, then broadcast to all ranks
            phase_t0 = time.time()
            batch = prepare_batch(rollouts, rewards, tokenizer, args.max_seq_len, "cpu")
            # Group-relative advantages: normalize within each prompt's
            # num_samples rollouts, not across the whole batch.
            advantages = compute_advantages(
                args.algorithm,
                batch["rewards"],
                num_samples_per_prompt=args.num_samples,
            )
            phase["pack_batch_s"] = time.time() - phase_t0

            print0(
                f"[step {step:04d}] prompts={len(examples)} rollouts={len(rollouts)} "
                f"fetch={phase['fetch_prompts_s']:.1f}s rollout={phase['rollout_s']:.1f}s "
                f"reward={phase['reward_s']:.1f}s pack={phase['pack_batch_s']:.1f}s "
                f"rewards[{reward_summary}]"
            )
        else:
            batch = None
            advantages = None
            mean_reward = 0.0
            reward_summary = ""

        phase_t0 = time.time()
        broadcast_result = broadcast_batch(
            master_process, ddp, device, ddp_rank, ddp_world_size, batch, advantages, mean_reward,
        )
        batch = broadcast_result.batch
        advantages = broadcast_result.advantages
        mean_reward = broadcast_result.mean_reward
        batch, advantages = local_shard(batch, advantages, ddp_rank, ddp_world_size)
        local_bsz = batch["input_ids"].shape[0]
        phase["broadcast_and_shard_s"] = time.time() - phase_t0

        # 5. Old log-probs. With ppo_epochs == 1, weights are frozen across the
        # single optimizer step, so old_logprobs ≡ logprobs.detach() per
        # microbatch — skip the duplicate no-grad pass. With >1 epochs we need
        # a genuinely frozen snapshot, chunked to fit long sequences.
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
            # Shape: [B, T-1] per-token log-probs of the frozen snapshot.
            old_logprobs = torch.cat(old_logprobs_chunks, dim=0)
        else:
            old_logprobs = None
        phase["old_logprobs_s"] = time.time() - phase_t0

        # 6. Policy update: ppo_epochs optimizer steps over the rollout batch,
        # each with grad accumulation across micro-batches.
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
                mb_old_lp = logprobs.detach() if old_logprobs is None else old_logprobs[start:end]
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

        # 7. Checkpoint + ask remote worker to reload so next-step rollouts are on-policy
        phase_t0 = time.time()
        if master_process:
            sync_root = args.rollout_sync_dir or os.path.join(args.save_dir, "rollout_sync")
            checkpoint_path = materialize_rollout_checkpoint(
                raw_model,
                sync_root=sync_root,
                slot_idx=step % 2,
                tokenizer_source=args.model,
            )
            remote_vllm_reload(args.rollout_worker_url, checkpoint_path)
        phase["sync_rollout_s"] = time.time() - phase_t0

        dt = time.time() - t0
        avg_loss = total_loss / max(args.ppo_epochs, 1)
        grad_norm_mean = mean(grad_norms) if grad_norms else 0.0
        grad_norm_max = max(grad_norms) if grad_norms else 0.0
        response_len_mean = mean(resp_lens) if master_process and resp_lens else 0.0
        truncation_ratio = (
            (n_truncated / len(resp_lens))
            if master_process and resp_lens
            else 0.0
        )
        train_elapsed_s = time.time() - train_start_time

        print0(
            f"step {step:04d}/{args.num_steps:04d} | loss: {avg_loss:.4f} | reward: {mean_reward:.4f} "
            f"| local_bsz: {local_bsz} | dt: {dt:.1f}s "
            f"| phases(fetch={phase.get('fetch_prompts_s', 0.0):.1f}s "
            f"rollout={phase.get('rollout_s', 0.0):.1f}s "
            f"reward={phase.get('reward_s', 0.0):.1f}s "
            f"pack={phase.get('pack_batch_s', 0.0):.1f}s "
            f"bcast={phase['broadcast_and_shard_s']:.1f}s "
            f"oldlp={phase['old_logprobs_s']:.1f}s "
            f"update={phase['update_s']:.1f}s "
            f"sync={phase['sync_rollout_s']:.1f}s)"
        )

        wandb_metrics = {
            "train/loss": avg_loss,
            "train/avg_reward": mean_reward,
            "train/grad_norm_mean": grad_norm_mean,
            "train/grad_norm_max": grad_norm_max,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/step_time_s": dt,
            "train/elapsed_s": train_elapsed_s,
            "train/local_batch_size": float(local_bsz),
            "train/response_len_mean": response_len_mean,
            "train/truncation_ratio": truncation_ratio,
            "timing/fetch_prompts_s": phase.get("fetch_prompts_s", 0.0),
            "timing/rollout_s": phase.get("rollout_s", 0.0),
            "timing/reward_s": phase.get("reward_s", 0.0),
            "timing/pack_batch_s": phase.get("pack_batch_s", 0.0),
            "timing/broadcast_and_shard_s": phase["broadcast_and_shard_s"],
            "timing/old_logprobs_s": phase["old_logprobs_s"],
            "timing/update_s": phase["update_s"],
            "timing/sync_rollout_s": phase["sync_rollout_s"],
        }
        wandb_logger.log_metrics(step=step, metrics=wandb_metrics)

        # 8. Evaluation
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            print0("Evaluating...")
            # TODO: pass@k evaluation hook

        # 9. Periodic checkpoint
        if args.save_every > 0 and (step + 1) % args.save_every == 0 and master_process:
            step_path = os.path.join(args.save_dir, f"step_{step+1:06d}")
            raw_model.save_pretrained(step_path, safe_serialization=True)
            tokenizer.save_pretrained(step_path)
            print0(f"[step {step:04d}] saved checkpoint to {step_path}")

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
