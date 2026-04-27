"""Download Weave traces for a run name and export Traces datamodel JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from loguru import logger

from nanorl.trace_download import WeaveTraceDownloader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Weave traces by run name.")
    parser.add_argument("--weave-project", type=str, required=True, help="Weave project, e.g. entity/project")
    parser.add_argument("--run-name", type=str, required=True, help="Run name used during training")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    return parser


def _write_output_json(output_path: str, records: list[dict[str, object]]) -> None:
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2, ensure_ascii=False)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logger.info(f"{args.weave_project=}")
    logger.info(f"{args.run_name=}")
    logger.info(f"{args.output=}")

    downloader = WeaveTraceDownloader(project_name=args.weave_project)
    traces = downloader.download_by_run_name(run_name=args.run_name)
    logger.info(f"{len(traces)=}")

    serialized_traces = [asdict(trace) for trace in traces]
    _write_output_json(output_path=args.output, records=serialized_traces)


if __name__ == "__main__":
    main()
