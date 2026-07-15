#!/usr/bin/env python3
"""Upload the universal-personalization entry point and start it over SSH keys."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def run(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("GAZE_HOST", "192.168.1.85"))
    parser.add_argument("--user", default=os.environ.get("GAZE_USER", "luxliang"))
    parser.add_argument("--remote-dir", default=os.environ.get(
        "GAZE_REMOTE_DIR", "/home/luxliang/gaze-personalization"))
    parser.add_argument("--data-dir", default=os.environ.get(
        "EYE_DATA_DIR", "/home/luxliang/datasets/EXPORT_PUPIL_ALL"))
    parser.add_argument("--checkpoint", default=os.environ.get("UNIVERSAL_CHECKPOINT"))
    parser.add_argument("--session", default="gaze-personalization")
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--split-strategy", choices=("chronological", "interleaved"),
                        default="interleaved")
    parser.add_argument("--replace", action="store_true",
                        help="Replace an existing tmux session with the same name")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint or UNIVERSAL_CHECKPOINT is required")

    root = Path(__file__).resolve().parent
    target = f"{args.user}@{args.host}"
    for filename in ("personalize_from_universal.py", "train_ours_two_stage.py"):
        run(["scp", str(root / filename), f"{target}:{args.remote_dir}/{filename}"])

    if args.replace:
        run(["ssh", target, f"tmux kill-session -t {shlex.quote(args.session)} 2>/dev/null || true"])

    training_command = " ".join([
        f"cd {shlex.quote(args.remote_dir)}",
        "&&",
        f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_device)}",
        "python3 -u personalize_from_universal.py",
        f"--data-dir {shlex.quote(args.data_dir)}",
        f"--checkpoint {shlex.quote(args.checkpoint)}",
        f"--split-strategy {shlex.quote(args.split_strategy)}",
        "2>&1",
        "|",
        "tee personalize_from_universal.log",
    ])
    remote_command = (
        f"tmux new-session -d -s {shlex.quote(args.session)} "
        f"{shlex.quote('bash -lc ' + shlex.quote(training_command))}"
    )
    run(["ssh", target, remote_command])
    run(["ssh", target, f"tmux has-session -t {shlex.quote(args.session)}"])
    print(f"Started tmux session {args.session!r} on {target}")
    print(f"Log: {args.remote_dir}/personalize_from_universal.log")


if __name__ == "__main__":
    main()
