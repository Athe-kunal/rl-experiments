"""Weights & Biases logging helpers for RL training metrics."""

from __future__ import annotations

from typing import Any, NamedTuple

import wandb


class _WandbProjectSpec(NamedTuple):
    """Parsed W&B project identifier."""

    entity: str | None
    project: str


def _parse_project_name(project_name: str) -> _WandbProjectSpec:
    """Parse ``project_name`` into ``entity`` + ``project`` parts.

    Accepts either:
      * ``project``
      * ``entity/project``
    """
    clean_name = project_name.strip()
    if "/" not in clean_name:
        return _WandbProjectSpec(entity=None, project=clean_name)
    entity, project = clean_name.split("/", maxsplit=1)
    entity = entity.strip() or None
    project = project.strip()
    return _WandbProjectSpec(entity=entity, project=project)


class WandbTrainingLogger:
    """Logs scalar RL training metrics to Weights & Biases."""

    def __init__(
        self,
        *,
        project_name: str,
        run_name: str,
        config: dict[str, Any],
        enabled: bool,
    ):
        self._enabled = bool(enabled and project_name and wandb is not None)
        self._run = None
        if not self._enabled:
            return

        spec = _parse_project_name(project_name)
        assert wandb is not None
        self._run = wandb.init(
            project=spec.project,
            entity=spec.entity,
            name=run_name,
            config=config,
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("timing/*", step_metric="train/step")
        wandb.define_metric("validation/*", step_metric="train/step")
        wandb.define_metric("rewards/*", step_metric="train/step")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def log_metrics(self, *, step: int, metrics: dict[str, float]) -> None:
        """Log training metrics keyed by scalar metric name."""
        if not self._enabled:
            return
        assert wandb is not None
        payload: dict[str, float | int] = {"train/step": step}
        payload.update(metrics)
        wandb.log(payload)

    def finish(self) -> None:
        """Close the active W&B run, if any."""
        if self._run is None:
            return
        self._run.finish()
        self._run = None
