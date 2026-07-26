"""Shared parsing for research validation suite configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_suite(value: str) -> tuple[str, Path, list[Path]]:
    try:
        name, payload = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected NAME=/benchmark/dir or NAME=/summary.json::/result/dir1,/result/dir2"
        ) from error
    if "::" in payload:
        summary, result_dirs = payload.split("::", 1)
        directories = [Path(item) for item in result_dirs.split(",") if item]
        if not directories:
            raise argparse.ArgumentTypeError("Suite must contain at least one result directory")
        return name, Path(summary), directories
    directory = Path(payload)
    return name, directory / "summary.json", [directory]


def load_config(path: Path) -> list[tuple[str, Path, list[Path]]]:
    payload = json.loads(path.read_text())
    return [
        (
            suite["name"],
            Path(suite["summary_path"]),
            [Path(value) for value in suite["result_dirs"]],
        )
        for suite in payload["suites"]
    ]
