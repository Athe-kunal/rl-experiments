"""Utilities to fetch Weave traces and convert them into datamodels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nanorl.datamodels import TraceCompletion, Traces

try:
    import weave as _weave
except ImportError:  # pragma: no cover - optional dependency.
    weave: Any | None = None
else:
    weave = _weave


def _require_weave() -> Any:
    if weave is None:
        raise ImportError("weave is required for trace downloading. Please install weave.")
    return weave


def _extract_prompt(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
    return ""


def _extract_completions(payload: dict[str, Any]) -> list[TraceCompletion]:
    completions_payload = payload.get("completions")
    if isinstance(completions_payload, list):
        return _extract_grouped_completions(completions_payload)
    return _extract_single_completion(payload)


def _extract_grouped_completions(completions_payload: list[Any]) -> list[TraceCompletion]:
    completions: list[TraceCompletion] = []
    for item in completions_payload:
        if not isinstance(item, dict):
            continue
        completion_text = item.get("assistant_response", "")
        if not isinstance(completion_text, str):
            completion_text = ""
        completion_metadata = dict(item)
        completions.append(TraceCompletion(completion=completion_text, metadata=completion_metadata))
    return completions


def _extract_single_completion(payload: dict[str, Any]) -> list[TraceCompletion]:
    completion_text = payload.get("assistant_response", "")
    if not isinstance(completion_text, str):
        completion_text = ""
    completion_metadata = {
        "verifiable_reward": payload.get("verifiable_reward"),
        "reward_for_training": payload.get("reward_for_training"),
        "reward_metadata": payload.get("reward_metadata"),
    }
    return [TraceCompletion(completion=completion_text, metadata=completion_metadata)]


def _to_trace_datamodel(payload: dict[str, Any]) -> Traces:
    return Traces(
        prompt=_extract_prompt(payload),
        completions=_extract_completions(payload),
        run_name=str(payload.get("run_name", "")),
        step=int(payload.get("step", -1)) if payload.get("step") is not None else -1,
        example_id=str(payload.get("example_id", "")),
    )


def _extract_call_output(call: Any) -> dict[str, Any] | None:
    output = getattr(call, "output", None)
    if isinstance(output, dict):
        return output
    if isinstance(call, dict):
        output = call.get("output")
        if isinstance(output, dict):
            return output
    return None


def _get_calls_from_client(client: Any) -> Iterable[Any]:
    if hasattr(client, "get_calls"):
        return client.get_calls()
    if hasattr(client, "calls"):
        return client.calls()
    if hasattr(client, "query_calls"):
        return client.query_calls()
    raise RuntimeError(
        "Unsupported Weave client API. Expected one of get_calls/calls/query_calls methods."
    )


class WeaveTraceDownloader:
    """Downloads traces from Weave and materializes ``Traces`` datamodels."""

    def __init__(self, project_name: str):
        self._project_name = project_name

    def download_by_run_name(self, run_name: str) -> list[Traces]:
        weave_module = _require_weave()
        client = weave_module.init(self._project_name)
        calls = _get_calls_from_client(client)
        payloads = self._collect_matching_payloads(calls=calls, run_name=run_name)
        return self._build_datamodels(payloads)

    def _collect_matching_payloads(self, calls: Iterable[Any], run_name: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for call in calls:
            payload = _extract_call_output(call)
            if payload is None:
                continue
            if payload.get("run_name") != run_name:
                continue
            payloads.append(payload)
        return payloads

    def _build_datamodels(self, payloads: list[dict[str, Any]]) -> list[Traces]:
        return [_to_trace_datamodel(payload) for payload in payloads]
