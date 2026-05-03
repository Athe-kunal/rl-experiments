"""
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo
"""

import sys

import ray
import torch
from dataclasses import dataclass
from typing import Any, List, Tuple

from skyrl.train.config import AlgorithmConfig, make_config
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg
from skyrl.train.entrypoints.main_base import BasePPOExp

from skyrl.train.generators.base import GeneratorInput, GeneratorOutput


@dataclass
class DAPOAlgorithmConfig(AlgorithmConfig):
    """Extended algorithm config with DAPO-specific overlong buffer settings."""

    overlong_buffer_len: int = 512
    overlong_buffer_penalty_factor: float = 1.0


DAPOConfig = make_config(algorithm_cls=DAPOAlgorithmConfig)


class DAPOTrainer(RayPPOTrainer):
    """
    Custom trainer for DAPO.

    Overrides the postprocess_generator_output method to additionally apply soft overlong punishment to rewards.
    """

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer_len
        overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer_penalty_factor
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]

        assert not isinstance(rewards[0], list), "we assume verifiable sequence level rewards here"

        response_lengths = [len(response) for response in response_ids]
        max_response_length = self.cfg.generator.sampling_params.max_generate_length

        for i, response_length in enumerate(response_lengths):
            max_exceed_length = max_response_length - overlong_buffer_len
            if response_length > max_exceed_length and response_length <= max_response_length:
                exceed_length = response_length - max_exceed_length
                penalty = exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor

                rewards[i] -= penalty
            elif response_length > max_response_length:
                rewards[i] = 0.0

        generator_output["rewards"] = rewards

        return super().postprocess_generator_output(generator_output, uids)

    def _extract_prompt_messages(self, prompt: Any) -> list[dict[str, str]]:
        """Keep prompt as role/content messages (system+user) for trajectory logging."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            messages: list[dict[str, str]] = []
            for msg in prompt:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(role, str) and isinstance(content, str):
                    messages.append({"role": role, "content": content})
            if messages:
                return messages
        return [{"role": "user", "content": str(prompt)}]

    def _init_weave(self) -> None:
        """Initialize weave once; subsequent calls are no-ops."""
        if getattr(self, "_weave_initialized", False):
            return
        try:
            import weave

            project = self.cfg.trainer.project_name
            if project:
                weave.init(project)
            self._weave_initialized = True
            self._weave = weave
        except Exception:
            self._weave_initialized = False
            self._weave = None

    def log_generation_trajectories(
        self,
        *,
        step: int,
        generator_input: GeneratorInput,
        generator_output: GeneratorOutput,
        uids: List[str],
    ) -> None:
        import json

        if "wandb" not in self.tracker.logger:
            return

        wandb = self.tracker.logger["wandb"]

        self._init_weave()
        weave = self._weave

        prompts = generator_input.get("prompts", [])
        response_ids = generator_output.get("response_ids", [])
        rewards = generator_output.get("rewards", [])
        if not prompts or not response_ids or not rewards:
            return

        trajectory_ids = generator_output.get("trajectory_ids")
        groups: dict[str, list[int]] = {}
        for i in range(len(response_ids)):
            if trajectory_ids is not None and i < len(trajectory_ids) and trajectory_ids[i] is not None:
                key = trajectory_ids[i].instance_id
            elif i < len(uids):
                key = uids[i]
            else:
                key = f"idx_{i}"
            groups.setdefault(key, []).append(i)

        for prompt_index, (group_id, idxs) in enumerate(groups.items()):
            prompt_idx = idxs[0] % len(prompts)
            prompt_messages = self._extract_prompt_messages(prompts[prompt_idx])
            prompt_messages_str = json.dumps(prompt_messages)
            completions = []
            table = wandb.Table(columns=["step", "group_id", "completion_index", "messages", "completion", "reward"])
            for completion_index, j in enumerate(idxs):
                completion_text = self.tokenizer.decode(response_ids[j], skip_special_tokens=True)
                reward_value = float(rewards[j]) if not isinstance(rewards[j], list) else float(sum(rewards[j]))
                completions.append(
                    {
                        "completion_index": completion_index,
                        "assistant_response": completion_text,
                        "verifiable_reward": reward_value,
                        "reward_for_training": reward_value,
                    }
                )
                table.add_data(step, group_id, completion_index, prompt_messages_str, completion_text, reward_value)

            wandb.log({f"trajectories/group_{group_id}": table}, step=step)
            if weave is not None:
                weave.publish(
                    {
                        "step": step,
                        "group_id": group_id,
                        "prompt_index": prompt_index,
                        "messages": prompt_messages,
                        "completions": completions,
                    }
                )


class DAPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return DAPOTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = DAPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = DAPOConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
