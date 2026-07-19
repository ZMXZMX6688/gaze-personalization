#!/usr/bin/env python3
"""Large-scale output-adapter benchmark for frozen universal gaze models.

The expensive video decoding and universal-model inference are cached once per
held-out subject. Repeated calibration experiments then run entirely on the
cached 3D predictions and targets.
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from personalize_from_universal import (
    AngularBiasAdapter,
    ClipRecord,
    SubjectClipSource,
    angles_to_vector,
    angular_errors_deg,
    checkpoint_sha256,
    discover_sids,
    fit_angular_bias,
    load_universal_model,
    parse_calibration_sizes,
    predict_records,
    resolve_device,
    split_calibration_pool,
    split_interleaved_calibration_pool,
    summarize_errors,
    universal_subject_split,
    vector_to_angles,
    wrap_angle,
)


METHODS = ("bias", "rotation", "affine")
PROTOCOLS = ("chronological", "interleaved")
SCALE_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
WIN_THRESHOLD_DEG = 1e-3


def parse_csv_choices(value: str, allowed: Sequence[str]) -> List[str]:
    choices = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(choices) - set(allowed))
    if not choices or unknown:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated choices from {allowed}; unknown={unknown}"
        )
    return list(dict.fromkeys(choices))


def stratified_sample_indices(count: int, sample_count: int, seed: int) -> List[int]:
    """Choose one random item from each evenly spaced temporal bin."""
    if sample_count <= 0 or sample_count > count:
        raise ValueError(f"Invalid sample_count={sample_count} for count={count}")
    if sample_count == count:
        return list(range(count))
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, count, sample_count + 1, dtype=np.int64)
    selected = [int(rng.integers(edges[i], edges[i + 1])) for i in range(sample_count)]
    if len(set(selected)) != sample_count:
        raise AssertionError("Stratified bins produced duplicate indices")
    return selected


def find_largest_feasible_split(
    records: Sequence[ClipRecord],
    calibration_pool_sizes: Sequence[int],
    gap_segments: int,
    protocol: str,
) -> Tuple[int, List[ClipRecord], List[ClipRecord], List[int], Dict[int, str]]:
    """Find the largest requested calibration pool that preserves evaluation data.

    Short recordings cannot always support the largest requested K while also
    preserving the segment embargo. Trying requested sizes from largest to
    smallest keeps one common evaluation split for every feasible K on a
    subject, without weakening the protocol or fabricating calibration clips.
    """
    if protocol == "chronological":
        split_function = split_calibration_pool
    elif protocol == "interleaved":
        split_function = split_interleaved_calibration_pool
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    candidate_sizes = sorted({int(size) for size in calibration_pool_sizes}, reverse=True)
    if not candidate_sizes or candidate_sizes[-1] <= 0:
        raise ValueError("Calibration pool sizes must be positive")

    failures: Dict[int, str] = {}
    for pool_size in candidate_sizes:
        try:
            calibration_pool, evaluation, calibration_segments = split_function(
                records, pool_size, gap_segments
            )
            if len(calibration_pool) < pool_size:
                raise ValueError(
                    f"Only {len(calibration_pool)} calibration clips are available"
                )
            return (
                pool_size,
                calibration_pool,
                evaluation,
                calibration_segments,
                failures,
            )
        except ValueError as exc:
            failures[pool_size] = str(exc)

    attempted = ", ".join(
        f"K={size}: {failures[size]}" for size in candidate_sizes
    )
    raise ValueError(f"No feasible {protocol} split ({attempted})")


def rotation_matrix_from_vector(rotation: torch.Tensor) -> torch.Tensor:
    """Convert an axis-angle rotation vector to a 3x3 matrix."""
    rotation = rotation.float()
    theta = torch.linalg.vector_norm(rotation)
    x, y, z = rotation.unbind()
    skew = torch.stack((
        torch.stack((x.new_zeros(()), -z, y)),
        torch.stack((z, x.new_zeros(()), -x)),
        torch.stack((-y, x, x.new_zeros(()))),
    ))
    # torch.sinc(x) = sin(pi*x)/(pi*x), including the stable x=0 limit.
    a = torch.sinc(theta / math.pi)
    b = 0.5 * torch.sinc(theta / (2.0 * math.pi)).square()
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    return identity + a * skew + b * (skew @ skew)


class RotationAdapter(nn.Module):
    """Three-parameter SO(3) adapter."""

    def __init__(self, max_rotation_deg: float = 10.0):
        super().__init__()
        self.rotation = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.max_rotation_rad = math.radians(max_rotation_deg)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        matrix = rotation_matrix_from_vector(self.rotation)
        return F.normalize(vectors.float() @ matrix.T, dim=-1)

    def clamp_(self) -> None:
        with torch.no_grad():
            norm = torch.linalg.vector_norm(self.rotation)
            if norm > self.max_rotation_rad:
                self.rotation.mul_(self.max_rotation_rad / norm)

    def regularization_loss(self) -> torch.Tensor:
        return (self.rotation / self.max_rotation_rad).square().mean()


class TangentAffineAdapter(nn.Module):
    """Six-parameter yaw/pitch affine adapter near the identity transform."""

    def __init__(self, max_bias_deg: float = 10.0, max_linear_delta: float = 0.25):
        super().__init__()
        self.linear_delta = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.max_bias_rad = math.radians(max_bias_deg)
        self.max_linear_delta = float(max_linear_delta)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        angles = vector_to_angles(vectors)
        matrix = torch.eye(2, dtype=angles.dtype, device=angles.device) + self.linear_delta
        adapted = angles @ matrix.T + self.bias
        adapted = torch.stack((wrap_angle(adapted[..., 0]), adapted[..., 1]), dim=-1)
        return angles_to_vector(adapted)

    def clamp_(self) -> None:
        with torch.no_grad():
            self.linear_delta.clamp_(-self.max_linear_delta, self.max_linear_delta)
            self.bias.clamp_(-self.max_bias_rad, self.max_bias_rad)

    def regularization_loss(self) -> torch.Tensor:
        linear = (self.linear_delta / self.max_linear_delta).square().mean()
        bias = (self.bias / self.max_bias_rad).square().mean()
        return 0.5 * (linear + bias)


def fit_parametric_adapter(
    adapter: nn.Module,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    learning_rate: float,
    regularization: float,
) -> nn.Module:
    predictions = F.normalize(predictions.detach().float().cpu(), dim=-1)
    targets = F.normalize(targets.detach().float().cpu(), dim=-1)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    for _ in range(steps):
        adapted = adapter(predictions)
        data_loss = F.smooth_l1_loss(adapted, targets, beta=0.02)
        prior_loss = regularization * adapter.regularization_loss()
        loss = data_loss + prior_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adapter.clamp_()
    adapter.eval()
    return adapter


def fit_adapter(
    method: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    learning_rate: float,
    regularization: float,
    max_bias_deg: float,
    max_linear_delta: float,
) -> nn.Module:
    if method == "bias":
        return fit_angular_bias(
            predictions,
            targets,
            steps=steps,
            learning_rate=learning_rate,
            regularization=regularization,
            max_bias_deg=max_bias_deg,
        )
    if method == "rotation":
        adapter = RotationAdapter(max_rotation_deg=max_bias_deg)
    elif method == "affine":
        adapter = TangentAffineAdapter(
            max_bias_deg=max_bias_deg,
            max_linear_delta=max_linear_delta,
        )
    else:
        raise ValueError(f"Unknown adapter method: {method}")
    return fit_parametric_adapter(
        adapter,
        predictions,
        targets,
        steps=steps,
        learning_rate=learning_rate,
        regularization=regularization,
    )


def snapshot_parameters(adapter: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in adapter.named_parameters()}


def set_parameter_scale(
    adapter: nn.Module,
    raw_parameters: Mapping[str, torch.Tensor],
    scale: float,
) -> None:
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            parameter.copy_(raw_parameters[name] * scale)


def serialize_parameters(adapter: nn.Module) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for name, parameter in adapter.named_parameters():
        values = parameter.detach().cpu()
        if name in {"bias", "rotation"}:
            values = torch.rad2deg(values)
            name = f"{name}_deg"
        payload[name] = values.tolist()
    return payload


def fit_guarded_adapter(
    method: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    learning_rate: float,
    regularization: float,
    max_bias_deg: float,
    max_linear_delta: float,
    validation_fraction: float,
    min_validation_gain_deg: float,
) -> Tuple[nn.Module, Dict[str, object]]:
    count = len(predictions)
    if count < 4:
        raise ValueError("Guarded calibration requires at least four samples")
    validation_count = max(2, int(math.ceil(count * validation_fraction)))
    validation_count = min(validation_count, count - 2)
    fit_count = count - validation_count
    fit_predictions = predictions[:fit_count]
    fit_targets = targets[:fit_count]
    validation_predictions = predictions[fit_count:]
    validation_targets = targets[fit_count:]

    adapter = fit_adapter(
        method,
        fit_predictions,
        fit_targets,
        steps=steps,
        learning_rate=learning_rate,
        regularization=regularization,
        max_bias_deg=max_bias_deg,
        max_linear_delta=max_linear_delta,
    )
    raw_parameters = snapshot_parameters(adapter)
    base_validation_mean = float(angular_errors_deg(
        validation_predictions, validation_targets
    ).mean())
    candidate_means = {}
    with torch.no_grad():
        for scale in SCALE_CANDIDATES:
            set_parameter_scale(adapter, raw_parameters, scale)
            candidate_means[scale] = float(angular_errors_deg(
                adapter(validation_predictions), validation_targets
            ).mean())
    best_scale = min(SCALE_CANDIDATES, key=lambda value: (candidate_means[value], value))
    validation_gain = base_validation_mean - candidate_means[best_scale]
    if validation_gain < min_validation_gain_deg:
        best_scale = 0.0
        validation_gain = 0.0
    set_parameter_scale(adapter, raw_parameters, best_scale)
    return adapter, {
        "fit_count": fit_count,
        "validation_count": validation_count,
        "selected_scale": best_scale,
        "validation_base_mean_deg": base_validation_mean,
        "validation_personalized_mean_deg": candidate_means[best_scale],
        "validation_gain_deg": validation_gain,
        "parameters": serialize_parameters(adapter),
    }


def records_to_indices(
    all_records: Sequence[ClipRecord],
    selected_records: Sequence[ClipRecord],
) -> List[int]:
    index = {record: position for position, record in enumerate(all_records)}
    return [index[record] for record in selected_records]


def load_or_create_cache(
    sid: str,
    source: SubjectClipSource,
    cache_dir: Path,
    cache_metadata: Mapping[str, object],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    refresh: bool,
) -> Dict[str, object]:
    cache_path = cache_dir / f"{sid}.pt"
    if cache_path.is_file() and not refresh:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != dict(cache_metadata):
            raise ValueError(f"Cache metadata mismatch for {cache_path}; use --refresh-cache")
        return payload

    predictions, targets = predict_records(
        model,
        source,
        source.records,
        device,
        batch_size,
        num_workers,
    )
    payload = {
        "metadata": dict(cache_metadata),
        "sid": sid,
        "records": [asdict(record) for record in source.records],
        "predictions": predictions,
        "targets": targets,
    }
    torch.save(payload, cache_path)
    return payload


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(
    rows: Sequence[Mapping[str, object]],
    expected_subjects: int = 0,
) -> List[Dict[str, object]]:
    summaries = []
    keys = sorted({(
        str(row["protocol"]), str(row["method"]), int(row["calibration_size"])
    ) for row in rows})
    for protocol, method, calibration_size in keys:
        selected = [row for row in rows if (
            row["protocol"] == protocol
            and row["method"] == method
            and row["calibration_size"] == calibration_size
        )]
        repeats = sorted({int(row["repeat"]) for row in selected})
        repeat_metrics = []
        for repeat in repeats:
            repeat_rows = [row for row in selected if row["repeat"] == repeat]
            repeat_metrics.append({
                "base": float(np.mean([row["base_mean_deg"] for row in repeat_rows])),
                "personalized": float(np.mean([
                    row["personalized_mean_deg"] for row in repeat_rows
                ])),
                "improvement": float(np.mean([
                    row["improvement_mean_deg"] for row in repeat_rows
                ])),
                "win_rate": float(np.mean([
                    row["improvement_mean_deg"] > WIN_THRESHOLD_DEG for row in repeat_rows
                ])),
                "activation_rate": float(np.mean([
                    row["adapter_scale"] > 0.0 for row in repeat_rows
                ])),
            })
        improvements = np.asarray([item["improvement"] for item in repeat_metrics])
        personalized = np.asarray([item["personalized"] for item in repeat_metrics])
        subject_count = len({row["sid"] for row in selected})
        summaries.append({
            "protocol": protocol,
            "method": method,
            "calibration_size": calibration_size,
            "subjects": subject_count,
            "subject_coverage_rate": (
                float(subject_count / expected_subjects) if expected_subjects else 1.0
            ),
            "repeats": len(repeats),
            "base_macro_mean_deg": float(np.mean([item["base"] for item in repeat_metrics])),
            "personalized_macro_mean_deg": float(personalized.mean()),
            "personalized_macro_std_deg": float(personalized.std(ddof=0)),
            "mean_improvement_deg": float(improvements.mean()),
            "improvement_std_deg": float(improvements.std(ddof=0)),
            "improvement_p05_deg": float(np.percentile(improvements, 5)),
            "improvement_p95_deg": float(np.percentile(improvements, 95)),
            "subject_win_rate": float(np.mean([item["win_rate"] for item in repeat_metrics])),
            "subject_win_threshold_deg": WIN_THRESHOLD_DEG,
            "adapter_activation_rate": float(np.mean([
                item["activation_rate"] for item in repeat_metrics
            ])),
        })
    return summaries


def aggregate_subject_results(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    summaries = []
    keys = sorted({(
        str(row["protocol"]), str(row["method"]), int(row["calibration_size"]), str(row["sid"])
    ) for row in rows})
    for protocol, method, calibration_size, sid in keys:
        selected = [row for row in rows if (
            row["protocol"] == protocol
            and row["method"] == method
            and row["calibration_size"] == calibration_size
            and row["sid"] == sid
        )]
        improvements = np.asarray([row["improvement_mean_deg"] for row in selected])
        summaries.append({
            "protocol": protocol,
            "method": method,
            "calibration_size": calibration_size,
            "sid": sid,
            "repeats": len(selected),
            "base_mean_deg": float(np.mean([row["base_mean_deg"] for row in selected])),
            "personalized_mean_deg": float(np.mean([
                row["personalized_mean_deg"] for row in selected
            ])),
            "mean_improvement_deg": float(improvements.mean()),
            "improvement_std_deg": float(improvements.std(ddof=0)),
            "positive_repeat_rate": float(np.mean(improvements > WIN_THRESHOLD_DEG)),
            "positive_repeat_threshold_deg": WIN_THRESHOLD_DEG,
            "adapter_activation_rate": float(np.mean([
                row["adapter_scale"] > 0.0 for row in selected
            ])),
        })
    return summaries


def mean_jsonable(values: Iterable[float]) -> float:
    return float(np.mean(list(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get(
        "EYE_DATA_DIR", "/home/luxliang/datasets/EXPORT_PUPIL_ALL")))
    parser.add_argument("--checkpoint", type=Path,
                        default=os.environ.get("UNIVERSAL_CHECKPOINT"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sids", default="",
                        help="Comma-separated held-out SIDs; default uses the universal test split")
    parser.add_argument("--split-json", type=Path, default=None,
                        help="Explicit train/val/test split JSON, optionally containing folds")
    parser.add_argument("--fold-index", type=int, default=None,
                        help="Fold index when --split-json contains folds")
    parser.add_argument("--methods", type=lambda value: parse_csv_choices(value, METHODS),
                        default=list(METHODS))
    parser.add_argument("--protocols", type=lambda value: parse_csv_choices(value, PROTOCOLS),
                        default=list(PROTOCOLS))
    parser.add_argument("--calibration-sizes", type=parse_calibration_sizes,
                        default=parse_calibration_sizes("5,10,20,50"))
    parser.add_argument("--calibration-pool-size", type=int, default=0,
                        help="Reserved calibration pool size; 0 uses max(calibration_sizes)")
    parser.add_argument(
        "--insufficient-data-policy",
        choices=("error", "adaptive"),
        default="error",
        help=(
            "How to handle short subjects: error preserves strict legacy behavior; "
            "adaptive backs off to the largest feasible requested K and records coverage"
        ),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gap-segments", type=int, default=1)
    parser.add_argument("--segment-min-len", type=int, default=120)
    parser.add_argument("--max-segments", type=int, default=60)
    parser.add_argument("--clips-per-segment", type=int, default=15)
    parser.add_argument("--max-eval-clips", type=int, default=0)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="CPU threads for tiny adapter optimization operations")
    parser.add_argument("--adapter-steps", type=int, default=200)
    parser.add_argument("--adapter-lr", type=float, default=0.05)
    parser.add_argument("--adapter-regularization", type=float, default=1e-3)
    parser.add_argument("--max-bias-deg", type=float, default=10.0)
    parser.add_argument("--max-linear-delta", type=float, default=0.25)
    parser.add_argument("--calibration-validation-fraction", type=float, default=0.25)
    parser.add_argument("--min-calibration-gain-deg", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None:
        parser.error("--checkpoint or UNIVERSAL_CHECKPOINT is required")
    args.checkpoint = Path(args.checkpoint)
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.calibration_pool_size < 0:
        parser.error("--calibration-pool-size must be non-negative")
    if args.gap_segments < 0:
        parser.error("--gap-segments must be non-negative")
    if not 0.0 < args.calibration_validation_fraction < 1.0:
        parser.error("--calibration-validation-fraction must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    all_sids = discover_sids(args.data_dir)
    train_sids, val_sids, default_test_sids = universal_subject_split(all_sids, args.seed)
    if args.split_json is not None:
        with args.split_json.open("r") as handle:
            split_payload = json.load(handle)
        if "folds" in split_payload:
            if args.fold_index is None:
                parser.error("--fold-index is required when --split-json contains folds")
            try:
                split_payload = split_payload["folds"][args.fold_index]
            except IndexError:
                parser.error(f"Invalid --fold-index {args.fold_index}")
        train_sids = list(split_payload["train_sids"])
        val_sids = list(split_payload["val_sids"])
        default_test_sids = list(split_payload["test_sids"])
    test_sids = [sid for sid in args.sids.split(",") if sid] or default_test_sids
    unknown = sorted(set(test_sids) - set(all_sids))
    if unknown:
        parser.error(f"Unknown SIDs: {unknown}")

    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    cache_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "data_dir": str(args.data_dir.resolve()),
        "clip_len": args.clip_len,
        "stride": args.stride,
        "img_size": args.img_size,
        "segment_min_len": args.segment_min_len,
        "max_segments": args.max_segments,
        "clips_per_segment": args.clips_per_segment,
        "full_span_valid": True,
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    sources = {}
    caches = {}
    missing_cache_sids = [sid for sid in test_sids if (
        args.refresh_cache or not (args.cache_dir / f"{sid}.pt").is_file()
    )]
    model = load_universal_model(args.checkpoint, device) if missing_cache_sids else None
    for sid in test_sids:
        source = SubjectClipSource(
            args.data_dir,
            sid,
            clip_len=args.clip_len,
            stride=args.stride,
            img_size=args.img_size,
            segment_min_len=args.segment_min_len,
            max_segments=args.max_segments,
            clips_per_segment=args.clips_per_segment,
            full_span_valid=True,
        )
        sources[sid] = source
        caches[sid] = load_or_create_cache(
            sid,
            source,
            args.cache_dir,
            cache_metadata,
            model,
            device,
            args.batch_size,
            args.num_workers,
            args.refresh_cache,
        )
        print(f"[cache] {sid}: {len(source.records)} indexed clips", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path("runs") / f"personalization-benchmark-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    max_calibration = args.calibration_pool_size or max(args.calibration_sizes)
    if max_calibration < max(args.calibration_sizes):
        parser.error("--calibration-pool-size cannot be smaller than calibration sizes")
    rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    split_manifest = {}

    for protocol_index, protocol in enumerate(args.protocols):
        split_manifest[protocol] = {}
        for sid_index, sid in enumerate(test_sids):
            payload = caches[sid]
            records = [ClipRecord(**record) for record in payload["records"]]
            predictions = payload["predictions"].float()
            targets = payload["targets"].float()
            if args.insufficient_data_policy == "adaptive" and not args.calibration_pool_size:
                candidate_pool_sizes = args.calibration_sizes
            else:
                candidate_pool_sizes = [max_calibration]
            try:
                (
                    selected_pool_size,
                    calibration_pool,
                    evaluation,
                    calibration_segments,
                    split_failures,
                ) = find_largest_feasible_split(
                    records,
                    candidate_pool_sizes,
                    args.gap_segments,
                    protocol,
                )
            except ValueError as exc:
                if args.insufficient_data_policy == "error":
                    raise
                reason = str(exc)
                split_manifest[protocol][sid] = {
                    "status": "skipped",
                    "indexed_clips": len(records),
                    "feasible_calibration_sizes": [],
                    "skipped_calibration_sizes": list(args.calibration_sizes),
                    "reason": reason,
                }
                coverage_rows.append({
                    "protocol": protocol,
                    "sid": sid,
                    "status": "skipped",
                    "indexed_clips": len(records),
                    "calibration_pool_clips": 0,
                    "evaluation_clips": 0,
                    "feasible_calibration_sizes": "",
                    "skipped_calibration_sizes": ",".join(
                        str(size) for size in args.calibration_sizes
                    ),
                    "reason": reason,
                })
                print(f"[skip] {protocol} {sid}: {reason}", flush=True)
                continue

            feasible_calibration_sizes = [
                size for size in args.calibration_sizes if size <= len(calibration_pool)
            ]
            skipped_calibration_sizes = [
                size for size in args.calibration_sizes if size not in feasible_calibration_sizes
            ]
            if args.max_eval_clips:
                positions = np.linspace(
                    0, len(evaluation) - 1, args.max_eval_clips, dtype=np.int64
                )
                evaluation = [evaluation[int(position)] for position in positions]
            calibration_indices = records_to_indices(records, calibration_pool)
            evaluation_indices = records_to_indices(records, evaluation)
            calibration_predictions = predictions[calibration_indices]
            calibration_targets = targets[calibration_indices]
            evaluation_predictions = predictions[evaluation_indices]
            evaluation_targets = targets[evaluation_indices]
            base_metrics = summarize_errors(
                angular_errors_deg(evaluation_predictions, evaluation_targets)
            )
            split_manifest[protocol][sid] = {
                "status": "evaluated",
                "indexed_clips": len(records),
                "selected_calibration_pool_size": selected_pool_size,
                "feasible_calibration_sizes": feasible_calibration_sizes,
                "skipped_calibration_sizes": skipped_calibration_sizes,
                "larger_pool_failures": split_failures,
                "calibration_pool": [asdict(record) for record in calibration_pool],
                "evaluation": [asdict(record) for record in evaluation],
                "calibration_segment_ids": calibration_segments,
                "evaluation_segment_ids": sorted({record.segment_id for record in evaluation}),
            }
            coverage_rows.append({
                "protocol": protocol,
                "sid": sid,
                "status": "evaluated",
                "indexed_clips": len(records),
                "calibration_pool_clips": len(calibration_pool),
                "evaluation_clips": base_metrics["n"],
                "feasible_calibration_sizes": ",".join(
                    str(size) for size in feasible_calibration_sizes
                ),
                "skipped_calibration_sizes": ",".join(
                    str(size) for size in skipped_calibration_sizes
                ),
                "reason": "",
            })

            for calibration_size in feasible_calibration_sizes:
                for repeat in range(args.repeats):
                    selection_seed = int(np.random.SeedSequence([
                        args.seed, protocol_index, sid_index, calibration_size, repeat
                    ]).generate_state(1)[0])
                    selected_positions = stratified_sample_indices(
                        len(calibration_pool), calibration_size, selection_seed
                    )
                    selected_predictions = calibration_predictions[selected_positions]
                    selected_targets = calibration_targets[selected_positions]
                    for method in args.methods:
                        adapter, guard = fit_guarded_adapter(
                            method,
                            selected_predictions,
                            selected_targets,
                            steps=args.adapter_steps,
                            learning_rate=args.adapter_lr,
                            regularization=args.adapter_regularization,
                            max_bias_deg=args.max_bias_deg,
                            max_linear_delta=args.max_linear_delta,
                            validation_fraction=args.calibration_validation_fraction,
                            min_validation_gain_deg=args.min_calibration_gain_deg,
                        )
                        with torch.no_grad():
                            personalized_predictions = adapter(evaluation_predictions)
                        personalized_metrics = summarize_errors(
                            angular_errors_deg(personalized_predictions, evaluation_targets)
                        )
                        rows.append({
                            "protocol": protocol,
                            "method": method,
                            "sid": sid,
                            "calibration_size": calibration_size,
                            "repeat": repeat,
                            "selection_seed": selection_seed,
                            "evaluation_clips": base_metrics["n"],
                            "calibration_fit_clips": guard["fit_count"],
                            "calibration_validation_clips": guard["validation_count"],
                            "adapter_scale": guard["selected_scale"],
                            "calibration_validation_gain_deg": guard["validation_gain_deg"],
                            "base_mean_deg": base_metrics["mean"],
                            "base_median_deg": base_metrics["median"],
                            "base_p90_deg": base_metrics["p90"],
                            "personalized_mean_deg": personalized_metrics["mean"],
                            "personalized_median_deg": personalized_metrics["median"],
                            "personalized_p90_deg": personalized_metrics["p90"],
                            "improvement_mean_deg": (
                                base_metrics["mean"] - personalized_metrics["mean"]
                            ),
                            "adapter_parameters_json": json.dumps(
                                guard["parameters"], separators=(",", ":")
                            ),
                        })
            print(
                f"[sweep] {protocol} {sid}: eval={base_metrics['n']} "
                f"K={feasible_calibration_sizes} skipped={skipped_calibration_sizes} "
                f"rows={len(feasible_calibration_sizes) * args.repeats * len(args.methods)}",
                flush=True,
            )

    summary_rows = aggregate_results(rows, expected_subjects=len(test_sids))
    subject_summary_rows = aggregate_subject_results(rows)
    config = vars(args).copy()
    for key in ("data_dir", "checkpoint", "cache_dir", "output_dir", "split_json"):
        value = output_dir if key == "output_dir" else config[key]
        if value is not None:
            config[key] = str(value)
    config.update({
        "checkpoint_sha256": checkpoint_hash,
        "universal_train_sids": train_sids,
        "universal_val_sids": val_sids,
        "held_out_test_sids": test_sids,
        "row_count": len(rows),
    })

    write_csv(output_dir / "results.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "subject_summary.csv", subject_summary_rows)
    write_csv(output_dir / "coverage.csv", coverage_rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump({"config": config, "summary": summary_rows}, handle, indent=2)
    with (output_dir / "split_manifest.json").open("w") as handle:
        json.dump(split_manifest, handle, indent=2)
    print(json.dumps({
        "output_dir": str(output_dir),
        "rows": len(rows),
        "best": sorted(
            summary_rows,
            key=lambda row: row["mean_improvement_deg"],
            reverse=True,
        )[:10],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
