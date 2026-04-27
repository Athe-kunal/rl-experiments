"""Public datamodels shared across NanoRL utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceCompletion:
    """Single completion plus associated metadata from a trace."""

    completion: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Traces:
    """Grouped prompt trace with all sampled completions and metadata."""

    prompt: str
    completions: list[TraceCompletion]
    run_name: str = ""
    step: int = -1
    example_id: str = ""
