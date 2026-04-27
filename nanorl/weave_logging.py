"""Weave tracing helpers for rollout trajectories."""

from __future__ import annotations

from typing import Any

try:
    import weave as _weave
except ImportError:  # pragma: no cover - optional dependency.
    weave: Any | None = None
else:
    weave = _weave


def _empty_system_prompt() -> str:
    return ""


def _resolve_user_prompt(example_prompt: Any, rollout_prompt: Any) -> str:
    """Best-effort user prompt resolution across example/rollout objects."""
    if isinstance(example_prompt, str) and example_prompt:
        return example_prompt
    if isinstance(rollout_prompt, str) and rollout_prompt:
        return rollout_prompt
    return ""


def _extract_system_prompt(example_meta: dict[str, Any]) -> str:
    system_prompt = example_meta.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        return _empty_system_prompt()
    return system_prompt


class WeaveTrajectoryLogger:
    """Logs grouped rollout reward traces to W&B Weave."""

    def __init__(self, project_name: str, enabled: bool):
        self._enabled = bool(enabled and project_name and weave is not None)
        if self._enabled:
            assert weave is not None
            weave.init(project_name)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def log_trajectories(
        self,
        step: int,
        examples: list[Any],
        responses: list[str],
        reward_infos: list[dict[str, Any]],
        verifier_rewards: list[float],
        training_rewards: list[float],
        *,
        num_samples_per_prompt: int | None = None,
        rollouts: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        if num_samples_per_prompt is not None and num_samples_per_prompt > 0:
            self._log_grouped_trajectories(
                step=step,
                expanded_examples=examples,
                responses=responses,
                reward_infos=reward_infos,
                verifier_rewards=verifier_rewards,
                training_rewards=training_rewards,
                num_samples_per_prompt=num_samples_per_prompt,
                rollouts=rollouts,
                max_new_tokens=max_new_tokens,
            )
            return

        # Fallback: one trace per (example, response) pair.
        for idx, (example, response, reward_info, verifier_reward, training_reward) in enumerate(
            zip(examples, responses, reward_infos, verifier_rewards, training_rewards)
        ):
            self._log_single_trajectory(
                step=step,
                index=idx,
                example=example,
                response=response,
                reward_info=reward_info,
                verifier_reward=verifier_reward,
                training_reward=training_reward,
            )

    @weave.op() if weave is not None else (lambda fn: fn)
    def _log_single_trajectory(
        self,
        step: int,
        index: int,
        example: Any,
        response: str,
        reward_info: dict[str, Any],
        verifier_reward: float,
        training_reward: float,
    ) -> dict[str, Any]:
        system_prompt = _extract_system_prompt(getattr(example, "meta", {}))
        user_prompt = _resolve_user_prompt(
            getattr(example, "prompt", None),
            getattr(example, "rollout_prompt", None),
        )
        return {
            "step": step,
            "trajectory_index": index,
            "example_id": getattr(example, "id", ""),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "assistant_response": response,
            "verifiable_reward": verifier_reward,
            "reward_for_training": training_reward,
            "reward_metadata": reward_info,
        }

    @weave.op() if weave is not None else (lambda fn: fn)
    def _log_grouped_trajectory(
        self,
        *,
        step: int,
        prompt_index: int,
        example: Any,
        completions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt = _extract_system_prompt(getattr(example, "meta", {}))
        user_prompt = _resolve_user_prompt(
            getattr(example, "prompt", None),
            getattr(example, "rollout_prompt", None),
        )
        return {
            "step": step,
            "prompt_index": prompt_index,
            "example_id": getattr(example, "id", ""),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # One entry per sampled completion for this prompt.
            "completions": completions,
        }

    def _log_grouped_trajectories(
        self,
        *,
        step: int,
        expanded_examples: list[Any],
        responses: list[str],
        reward_infos: list[dict[str, Any]],
        verifier_rewards: list[float],
        training_rewards: list[float],
        num_samples_per_prompt: int,
        rollouts: list[dict[str, Any]] | None,
        max_new_tokens: int | None,
    ) -> None:
        if not expanded_examples:
            return
        n = len(responses)
        group_size = num_samples_per_prompt
        n_groups = n // group_size
        for prompt_index in range(n_groups):
            start = prompt_index * group_size
            end = start + group_size
            example = expanded_examples[start]
            completions: list[dict[str, Any]] = []
            for j in range(start, end):
                rollout = rollouts[j] if rollouts is not None and j < len(rollouts) else None
                response_ids = rollout.get("response_ids") if isinstance(rollout, dict) else None
                prompt_ids = rollout.get("prompt_ids") if isinstance(rollout, dict) else None
                response_len = len(response_ids) if isinstance(response_ids, list) else None
                prompt_len = len(prompt_ids) if isinstance(prompt_ids, list) else None
                truncated = (
                    bool(max_new_tokens is not None and response_len is not None and response_len >= max_new_tokens)
                    if max_new_tokens is not None
                    else None
                )
                completions.append(
                    {
                        "completion_index": j - start,
                        "global_rollout_index": j,
                        "rollout_id": f"s{step}_p{prompt_index}_k{j - start}",
                        "assistant_response": responses[j],
                        "verifiable_reward": verifier_rewards[j],
                        "reward_for_training": training_rewards[j],
                        "reward_metadata": reward_infos[j],
                        "rollout": {
                            "prompt": rollout.get("prompt") if isinstance(rollout, dict) else None,
                            "prompt_ids": prompt_ids,
                            "response_ids": response_ids,
                            "prompt_len": prompt_len,
                            "response_len": response_len,
                            "truncated": truncated,
                        },
                    }
                )
            self._log_grouped_trajectory(
                step=step,
                prompt_index=prompt_index,
                example=example,
                completions=completions,
            )
