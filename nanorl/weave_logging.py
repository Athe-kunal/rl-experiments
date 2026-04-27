"""Weave tracing helpers for rollout trajectories."""

from __future__ import annotations

from typing import Any

try:
    import weave
except ImportError:  # pragma: no cover - optional dependency.
    weave = None


def _empty_system_prompt() -> str:
    return ""


def _resolve_user_prompt(example_prompt: str, rollout_prompt: str) -> str:
    if example_prompt:
        return example_prompt
    return rollout_prompt


def _extract_system_prompt(example_meta: dict[str, Any]) -> str:
    system_prompt = example_meta.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        return _empty_system_prompt()
    return system_prompt


class WeaveTrajectoryLogger:
    """Logs per-trajectory reward traces to W&B Weave."""

    def __init__(self, project_name: str, enabled: bool):
        self._enabled = bool(enabled and project_name and weave is not None)
        if self._enabled:
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
    ) -> None:
        if not self._enabled:
            return
        for idx, (example, response, reward_info, verifier_reward, training_reward) in enumerate(zip(
            examples, responses, reward_infos, verifier_rewards, training_rewards,
        )):
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
        user_prompt = _resolve_user_prompt(getattr(example, "prompt", ""), getattr(example, "prompt", ""))
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
