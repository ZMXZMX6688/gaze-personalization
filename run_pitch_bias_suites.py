#!/usr/bin/env python3
"""Replay audited benchmark suites with the frozen one-parameter pitch adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from research_suite_config import load_config


VALUE_OPTIONS = {
    "data_dir": "--data-dir",
    "checkpoint": "--checkpoint",
    "cache_dir": "--cache-dir",
    "device": "--device",
    "fold_index": "--fold-index",
    "calibration_pool_size": "--calibration-pool-size",
    "repeats": "--repeats",
    "gap_segments": "--gap-segments",
    "segment_min_len": "--segment-min-len",
    "max_segments": "--max-segments",
    "clips_per_segment": "--clips-per-segment",
    "max_eval_clips": "--max-eval-clips",
    "clip_len": "--clip-len",
    "stride": "--stride",
    "img_size": "--img-size",
    "batch_size": "--batch-size",
    "num_workers": "--num-workers",
    "torch_threads": "--torch-threads",
    "adapter_steps": "--adapter-steps",
    "adapter_lr": "--adapter-lr",
    "adapter_regularization": "--adapter-regularization",
    "max_bias_deg": "--max-bias-deg",
    "max_linear_delta": "--max-linear-delta",
    "calibration_validation_fraction": "--calibration-validation-fraction",
    "min_calibration_gain_deg": "--min-calibration-gain-deg",
    "gate_strategy": "--gate-strategy",
    "seed": "--seed",
}


def replay_command(config: dict, output_dir: Path) -> list[str]:
    command = [sys.executable, "personalization_benchmark.py"]
    for key, option in VALUE_OPTIONS.items():
        value = config.get(key)
        if value is not None:
            command.extend((option, str(value)))
    if config.get("split_json"):
        command.extend(("--split-json", str(config["split_json"])))
    if config.get("sids"):
        command.extend(("--sids", str(config["sids"])))
    command.extend(("--methods", "pitch_bias"))
    command.extend(("--protocols", ",".join(config["protocols"])))
    command.extend(
        ("--calibration-sizes", ",".join(map(str, config["calibration_sizes"])))
    )
    command.extend(("--output-dir", str(output_dir)))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = []
    for suite, _, result_dirs in load_config(args.config):
        for fold, source_dir in enumerate(result_dirs):
            config = json.loads((source_dir / "summary.json").read_text())["config"]
            output_dir = args.output_root / suite / f"fold{fold}"
            command = replay_command(config, output_dir)
            print(f"[replay] {suite}/fold{fold}", flush=True)
            subprocess.run(command, check=True)
            manifest.append(
                {
                    "suite": suite,
                    "fold": fold,
                    "source_dir": str(source_dir),
                    "output_dir": str(output_dir),
                    "command": command,
                }
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "replay_manifest.json").write_text(
        json.dumps({"runs": manifest}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
