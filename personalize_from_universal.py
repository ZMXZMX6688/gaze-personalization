#!/usr/bin/env python3
"""Calibrate small per-user gaze adapters on top of a frozen universal model.

The universal checkpoint is never updated. Each held-out subject uses labeled,
segment-disjoint clips for calibration and evaluation.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_video

from train_ours_two_stage import (
    EYE_CLIP_FPS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ResNet18GRUModel,
    detect_concat_points,
    load_gaze_vec,
    load_validity,
)


@dataclass(frozen=True)
class ClipRecord:
    sid: str
    start: int
    target_frame: int
    segment_id: int
    segment_start: int
    segment_end: int


def evenly_select(items: Sequence, count: int) -> List:
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    positions = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(position)] for position in positions]


def validate_sequential_frame_ids(path: Path) -> None:
    """Reject files where permissive row parsing would shift label alignment."""
    expected = 1
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        next(handle, None)
        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n").split(";")
            try:
                frame_id = int(parts[0])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Malformed frame id at {path}:{line_number}") from exc
            if frame_id != expected:
                raise ValueError(
                    f"Non-sequential frame id at {path}:{line_number}: "
                    f"expected {expected}, got {frame_id}"
                )
            expected += 1


def discover_sids(data_dir: Path) -> List[str]:
    pattern = re.compile(r"^(NVIDIA(?:AR|VR)_\d+_1)\.mp4pupil_seg_3D\.mp4$")
    return sorted({
        match.group(1)
        for path in data_dir.iterdir()
        if path.is_file() and (match := pattern.match(path.name))
    })


def universal_subject_split(sids: Sequence[str], seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Reproduce the split used by train_ours_two_stage.py."""
    rng = random.Random(seed)
    shuffled = list(sids)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * 0.1)))
    test_sids = shuffled[-n_test:]
    train_val_sids = shuffled[:-n_test]
    rng.shuffle(train_val_sids)
    n_val = max(1, int(round(len(train_val_sids) * 0.1)))
    return train_val_sids[n_val:], train_val_sids[:n_val], test_sids


class SubjectClipSource:
    """Index and decode non-crossing clips for one subject."""

    def __init__(
        self,
        data_dir: Path,
        sid: str,
        clip_len: int = 8,
        stride: int = 4,
        img_size: int = 240,
        segment_min_len: int = 120,
        max_segments: int = 60,
        clips_per_segment: int = 15,
        full_span_valid: bool = True,
    ):
        self.data_dir = data_dir
        self.sid = sid
        self.clip_len = clip_len
        self.stride = stride
        self.img_size = img_size
        self.need_frames = (clip_len - 1) * stride + 1
        self.video_path = data_dir / f"{sid}.mp4pupil_seg_3D.mp4"
        self.gaze_path = data_dir / f"{sid}.mp4gaze_vec.txt"
        self.validity_path = data_dir / f"{sid}.mp4validity_pupil.txt"
        self.landmark_path = data_dir / f"{sid}.mp4pupil_lm_3D.txt"
        required = (self.video_path, self.gaze_path, self.validity_path, self.landmark_path)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing files for {sid}: {missing}")

        for path in (self.gaze_path, self.validity_path, self.landmark_path):
            validate_sequential_frame_ids(path)

        gaze = load_gaze_vec(self.gaze_path)
        valid = load_validity(self.validity_path).astype(np.int32)
        n = min(len(gaze), len(valid))
        self.gaze = gaze[:n]
        self.valid = valid[:n]
        gaze_norm = np.linalg.norm(self.gaze, axis=1)
        bad_gaze = (
            np.all(self.gaze == -1.0, axis=1)
            | ~np.all(np.isfinite(self.gaze), axis=1)
            | (gaze_norm < 1e-8)
        )
        self.valid[bad_gaze] = 0

        _, raw_segments = detect_concat_points(
            data_dir, sid, z_thresh=10.0, min_gap=300, fps=EYE_CLIP_FPS
        )
        eligible = [
            (segment_id, segment)
            for segment_id, segment in enumerate(raw_segments)
            if segment[2] >= segment_min_len
        ]
        eligible = evenly_select(eligible, max_segments) if max_segments else eligible

        records: List[ClipRecord] = []
        for segment_id, (start, end, _, _) in eligible:
            valid_starts = []
            for clip_start in range(start, end - self.need_frames + 1, stride):
                target_frame = clip_start + self.need_frames - 1
                if target_frame >= n:
                    continue
                if full_span_valid:
                    is_valid = bool(np.all(self.valid[clip_start:target_frame + 1] == 1))
                else:
                    indices = np.arange(clip_start, target_frame + 1, stride, dtype=np.int64)
                    is_valid = bool(np.all(self.valid[indices] == 1))
                if is_valid:
                    valid_starts.append(clip_start)
            for clip_start in evenly_select(valid_starts, clips_per_segment):
                records.append(ClipRecord(
                    sid=sid,
                    start=int(clip_start),
                    target_frame=int(clip_start + self.need_frames - 1),
                    segment_id=int(segment_id),
                    segment_start=int(start),
                    segment_end=int(end),
                ))
        self.records = sorted(records, key=lambda record: (record.segment_id, record.start))
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def decode(self, record: ClipRecord) -> Tuple[torch.Tensor, torch.Tensor]:
        start_sec = record.start / EYE_CLIP_FPS
        end_sec = (record.start + self.need_frames) / EYE_CLIP_FPS
        frames, _, _ = read_video(
            str(self.video_path), start_pts=start_sec, end_pts=end_sec, pts_unit="sec"
        )
        if frames.shape[0] == 0:
            raise RuntimeError(f"Decoded zero frames for {self.sid} at frame {record.start}")
        if frames.shape[0] < self.need_frames:
            padding = frames[-1:].repeat(self.need_frames - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, padding])
        elif frames.shape[0] > self.need_frames:
            frames = frames[:self.need_frames]
        frames = frames[::self.stride]
        if frames.shape[0] != self.clip_len:
            raise RuntimeError(
                f"Expected {self.clip_len} sampled frames, got {frames.shape[0]} "
                f"for {self.sid} at {record.start}"
            )
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        frames = F.interpolate(
            frames, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        frames = self.normalize(frames)
        target = torch.from_numpy(self.gaze[record.target_frame]).float()
        return frames, target


class RecordDataset(Dataset):
    def __init__(self, source: SubjectClipSource, records: Sequence[ClipRecord]):
        self.source = source
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return self.source.decode(self.records[index])


def split_calibration_pool(
    records: Sequence[ClipRecord],
    max_calibration_clips: int,
    gap_segments: int,
) -> Tuple[List[ClipRecord], List[ClipRecord], List[int]]:
    """Use early whole segments for calibration and strictly later segments for evaluation."""
    grouped: List[Tuple[int, List[ClipRecord]]] = []
    for record in records:
        if not grouped or grouped[-1][0] != record.segment_id:
            grouped.append((record.segment_id, []))
        grouped[-1][1].append(record)
    if not grouped:
        raise ValueError("No indexed clips")

    pool_candidates: List[ClipRecord] = []
    last_pool_group = -1
    for group_index, (_, group_records) in enumerate(grouped):
        pool_candidates.extend(group_records)
        last_pool_group = group_index
        if len(pool_candidates) >= max_calibration_clips:
            break
    calibration_pool = evenly_select(pool_candidates, max_calibration_clips)
    evaluation_start = last_pool_group + 1 + gap_segments
    evaluation = [record for _, group in grouped[evaluation_start:] for record in group]
    if not calibration_pool:
        raise ValueError("Calibration pool is empty")
    if not evaluation:
        raise ValueError("Evaluation split is empty after calibration and gap segments")

    calibration_segment_ids = sorted({record.segment_id for record in pool_candidates})
    evaluation_segment_ids = {record.segment_id for record in evaluation}
    if evaluation_segment_ids.intersection(calibration_segment_ids):
        raise AssertionError("Calibration and evaluation share a segment")
    if max(record.target_frame for record in pool_candidates) >= min(record.start for record in evaluation):
        raise AssertionError("Calibration is not chronologically before evaluation")
    return calibration_pool, evaluation, calibration_segment_ids


def split_interleaved_calibration_pool(
    records: Sequence[ClipRecord],
    max_calibration_clips: int,
    gap_segments: int,
) -> Tuple[List[ClipRecord], List[ClipRecord], List[int]]:
    """Uniformly reserve calibration segments and evaluate on all other non-neighbors."""
    grouped: List[Tuple[int, List[ClipRecord]]] = []
    for record in records:
        if not grouped or grouped[-1][0] != record.segment_id:
            grouped.append((record.segment_id, []))
        grouped[-1][1].append(record)
    if not grouped:
        raise ValueError("No indexed clips")

    selected_group_indices = []
    pool_candidates: List[ClipRecord] = []
    for group_count in range(1, len(grouped) + 1):
        positions = np.linspace(0, len(grouped) - 1, group_count, dtype=np.int64)
        candidate_indices = sorted({int(position) for position in positions})
        candidates = [record for index in candidate_indices for record in grouped[index][1]]
        if len(candidates) >= max_calibration_clips:
            selected_group_indices = candidate_indices
            pool_candidates = candidates
            break
    if len(pool_candidates) < max_calibration_clips:
        raise ValueError("Not enough clips for the requested interleaved calibration pool")

    calibration_pool = evenly_select(pool_candidates, max_calibration_clips)
    excluded_group_indices = set()
    for group_index in selected_group_indices:
        lower = max(0, group_index - gap_segments)
        upper = min(len(grouped), group_index + gap_segments + 1)
        excluded_group_indices.update(range(lower, upper))
    evaluation = [
        record
        for group_index, (_, group_records) in enumerate(grouped)
        if group_index not in excluded_group_indices
        for record in group_records
    ]
    if not evaluation:
        raise ValueError("Evaluation split is empty after interleaved calibration embargoes")

    calibration_segment_ids = sorted({record.segment_id for record in pool_candidates})
    evaluation_segment_ids = {record.segment_id for record in evaluation}
    if evaluation_segment_ids.intersection(calibration_segment_ids):
        raise AssertionError("Calibration and evaluation share a segment")
    return calibration_pool, evaluation, calibration_segment_ids


def vector_to_angles(vectors: torch.Tensor) -> torch.Tensor:
    vectors = F.normalize(vectors.float(), dim=-1)
    yaw = torch.atan2(vectors[..., 0], vectors[..., 2])
    horizontal = torch.sqrt(vectors[..., 0].square() + vectors[..., 2].square()).clamp_min(1e-8)
    pitch = torch.atan2(vectors[..., 1], horizontal)
    return torch.stack((yaw, pitch), dim=-1)


def angles_to_vector(angles: torch.Tensor) -> torch.Tensor:
    yaw, pitch = angles.unbind(dim=-1)
    cos_pitch = torch.cos(pitch)
    return torch.stack((
        torch.sin(yaw) * cos_pitch,
        torch.sin(pitch),
        torch.cos(yaw) * cos_pitch,
    ), dim=-1)


class AngularBiasAdapter(nn.Module):
    """Two-parameter user adapter in yaw/pitch tangent coordinates."""

    def __init__(self, max_bias_deg: float = 10.0):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.max_bias_rad = math.radians(max_bias_deg)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return angles_to_vector(vector_to_angles(vectors) + self.bias)

    def clamp_(self) -> None:
        with torch.no_grad():
            self.bias.clamp_(-self.max_bias_rad, self.max_bias_rad)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def fit_angular_bias(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 250,
    learning_rate: float = 0.05,
    regularization: float = 1e-3,
    max_bias_deg: float = 10.0,
) -> AngularBiasAdapter:
    predictions = F.normalize(predictions.detach().float().cpu(), dim=-1)
    targets = F.normalize(targets.detach().float().cpu(), dim=-1)
    adapter = AngularBiasAdapter(max_bias_deg=max_bias_deg)

    residual = vector_to_angles(targets) - vector_to_angles(predictions)
    residual[:, 0] = wrap_angle(residual[:, 0])
    with torch.no_grad():
        adapter.bias.copy_(residual.median(dim=0).values)
        adapter.clamp_()

    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    for _ in range(steps):
        adapted = adapter(predictions)
        data_loss = F.smooth_l1_loss(adapted, targets, beta=0.02)
        prior_loss = regularization * (adapter.bias / adapter.max_bias_rad).square().mean()
        loss = data_loss + prior_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adapter.clamp_()
    adapter.eval()
    return adapter


def fit_guarded_angular_bias(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 250,
    learning_rate: float = 0.05,
    regularization: float = 1e-3,
    max_bias_deg: float = 10.0,
    validation_fraction: float = 0.25,
    min_validation_gain_deg: float = 0.02,
) -> Tuple[AngularBiasAdapter, Dict[str, float]]:
    """Fit on early calibration samples and select shrinkage on later calibration samples."""
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

    adapter = fit_angular_bias(
        fit_predictions,
        fit_targets,
        steps=steps,
        learning_rate=learning_rate,
        regularization=regularization,
        max_bias_deg=max_bias_deg,
    )
    raw_bias = adapter.bias.detach().clone()
    base_validation_mean = float(angular_errors_deg(
        validation_predictions, validation_targets
    ).mean())
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    candidate_means = {}
    with torch.no_grad():
        for scale in candidates:
            adapter.bias.copy_(raw_bias * scale)
            mean_error = float(angular_errors_deg(
                adapter(validation_predictions), validation_targets
            ).mean())
            candidate_means[scale] = mean_error
    best_scale = min(candidates, key=lambda scale: (candidate_means[scale], scale))
    validation_gain = base_validation_mean - candidate_means[best_scale]
    if validation_gain < min_validation_gain_deg:
        best_scale = 0.0
        validation_gain = 0.0
    with torch.no_grad():
        adapter.bias.copy_(raw_bias * best_scale)
    return adapter, {
        "fit_count": fit_count,
        "validation_count": validation_count,
        "selected_scale": best_scale,
        "validation_base_mean_deg": base_validation_mean,
        "validation_personalized_mean_deg": candidate_means[best_scale],
        "validation_gain_deg": validation_gain,
    }


def angular_errors_deg(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    predictions = F.normalize(predictions.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    cosine = (predictions * targets).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine)).detach().cpu().numpy()


def summarize_errors(errors: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(errors), dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "n": int(values.size),
    }


@torch.inference_mode()
def predict_records(
    model: nn.Module,
    source: SubjectClipSource,
    records: Sequence[ClipRecord],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        RecordDataset(source, records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    targets = []
    for frames, target in loader:
        frames = frames.to(device, non_blocking=True)
        prediction = F.normalize(model(frames), dim=-1).cpu()
        predictions.append(prediction)
        targets.append(F.normalize(target.float(), dim=-1))
    return torch.cat(predictions), torch.cat(targets)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_universal_model(checkpoint: Path, device: torch.device) -> ResNet18GRUModel:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = ResNet18GRUModel(personalize_mode="none", pretrained_backbone=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def parse_calibration_sizes(value: str) -> List[int]:
    sizes = sorted({int(part) for part in value.split(",") if part.strip()})
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Calibration sizes must be positive comma-separated integers")
    return sizes


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get(
        "EYE_DATA_DIR", "/home/luxliang/datasets/EXPORT_PUPIL_ALL")))
    parser.add_argument("--checkpoint", type=Path,
                        default=os.environ.get("UNIVERSAL_CHECKPOINT"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sids", default="",
                        help="Comma-separated held-out SIDs; default reproduces universal test split")
    parser.add_argument("--calibration-sizes", type=parse_calibration_sizes,
                        default=parse_calibration_sizes("5,10,20,50"))
    parser.add_argument("--gap-segments", type=int, default=1)
    parser.add_argument("--split-strategy", choices=("chronological", "interleaved"),
                        default="chronological")
    parser.add_argument("--segment-min-len", type=int, default=120)
    parser.add_argument("--max-segments", type=int, default=60)
    parser.add_argument("--clips-per-segment", type=int, default=15)
    parser.add_argument("--max-eval-clips", type=int, default=0)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--adapter-steps", type=int, default=250)
    parser.add_argument("--adapter-lr", type=float, default=0.05)
    parser.add_argument("--adapter-regularization", type=float, default=1e-3)
    parser.add_argument("--max-bias-deg", type=float, default=10.0)
    parser.add_argument("--calibration-validation-fraction", type=float, default=0.25)
    parser.add_argument("--min-calibration-gain-deg", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gap_segments < 0:
        parser.error("--gap-segments must be non-negative")
    if not 0.0 < args.calibration_validation_fraction < 1.0:
        parser.error("--calibration-validation-fraction must be between 0 and 1")
    if args.min_calibration_gain_deg < 0.0:
        parser.error("--min-calibration-gain-deg must be non-negative")
    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    if not args.index_only and args.checkpoint is None:
        parser.error("--checkpoint or UNIVERSAL_CHECKPOINT is required")
    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)
        if not args.index_only and not args.checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    all_sids = discover_sids(args.data_dir)
    train_sids, val_sids, default_test_sids = universal_subject_split(all_sids, args.seed)
    test_sids = [sid for sid in args.sids.split(",") if sid] or default_test_sids
    unknown = sorted(set(test_sids) - set(all_sids))
    if unknown:
        parser.error(f"Unknown SIDs: {unknown}")

    max_calibration = max(args.calibration_sizes)
    sources = {}
    split_manifest = {}
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
        split_function = (split_calibration_pool if args.split_strategy == "chronological"
                          else split_interleaved_calibration_pool)
        calibration_pool, evaluation, calibration_segments = split_function(
            source.records, max_calibration, args.gap_segments)
        if args.max_eval_clips:
            evaluation = evenly_select(evaluation, args.max_eval_clips)
        sources[sid] = source
        split_manifest[sid] = {
            "indexed_clips": len(source.records),
            "calibration_pool": [asdict(record) for record in calibration_pool],
            "evaluation": [asdict(record) for record in evaluation],
            "calibration_segment_ids": calibration_segments,
            "evaluation_segment_ids": sorted({record.segment_id for record in evaluation}),
        }
        print(
            f"[index] {sid}: total={len(source.records)} "
            f"calibration_pool={len(calibration_pool)} evaluation={len(evaluation)}",
            flush=True,
        )

    if args.index_only:
        print(json.dumps({
            "train_sids": train_sids,
            "val_sids": val_sids,
            "test_sids": test_sids,
            "splits": split_manifest,
        }, indent=2))
        return

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path("runs") / f"personalization-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    model = load_universal_model(args.checkpoint, device)
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    print(f"[model] frozen universal checkpoint on {device}: {checkpoint_hash}", flush=True)

    rows = []
    adapter_payload = {}
    for sid in test_sids:
        source = sources[sid]
        manifest = split_manifest[sid]
        calibration_pool = [ClipRecord(**record) for record in manifest["calibration_pool"]]
        evaluation = [ClipRecord(**record) for record in manifest["evaluation"]]
        calibration_predictions, calibration_targets = predict_records(
            model, source, calibration_pool, device, args.batch_size, args.num_workers
        )
        evaluation_predictions, evaluation_targets = predict_records(
            model, source, evaluation, device, args.batch_size, args.num_workers
        )
        base_metrics = summarize_errors(
            angular_errors_deg(evaluation_predictions, evaluation_targets)
        )
        adapter_payload[sid] = {}
        for calibration_size in args.calibration_sizes:
            selected_indices = evenly_select(list(range(len(calibration_pool))), calibration_size)
            adapter, guard = fit_guarded_angular_bias(
                calibration_predictions[selected_indices],
                calibration_targets[selected_indices],
                steps=args.adapter_steps,
                learning_rate=args.adapter_lr,
                regularization=args.adapter_regularization,
                max_bias_deg=args.max_bias_deg,
                validation_fraction=args.calibration_validation_fraction,
                min_validation_gain_deg=args.min_calibration_gain_deg,
            )
            with torch.no_grad():
                personalized_predictions = adapter(evaluation_predictions)
            personalized_metrics = summarize_errors(
                angular_errors_deg(personalized_predictions, evaluation_targets)
            )
            bias_deg = torch.rad2deg(adapter.bias.detach()).tolist()
            row = {
                "sid": sid,
                "calibration_size": calibration_size,
                "evaluation_clips": base_metrics["n"],
                "yaw_bias_deg": bias_deg[0],
                "pitch_bias_deg": bias_deg[1],
                "adapter_scale": guard["selected_scale"],
                "calibration_fit_clips": guard["fit_count"],
                "calibration_validation_clips": guard["validation_count"],
                "calibration_validation_gain_deg": guard["validation_gain_deg"],
                "base_mean_deg": base_metrics["mean"],
                "base_median_deg": base_metrics["median"],
                "base_p90_deg": base_metrics["p90"],
                "personalized_mean_deg": personalized_metrics["mean"],
                "personalized_median_deg": personalized_metrics["median"],
                "personalized_p90_deg": personalized_metrics["p90"],
                "improvement_mean_deg": base_metrics["mean"] - personalized_metrics["mean"],
            }
            rows.append(row)
            adapter_payload[sid][str(calibration_size)] = {
                "yaw_bias_deg": bias_deg[0],
                "pitch_bias_deg": bias_deg[1],
                **guard,
            }
            print(
                f"[result] {sid} K={calibration_size}: "
                f"{base_metrics['mean']:.3f} -> {personalized_metrics['mean']:.3f} deg "
                f"(delta={row['improvement_mean_deg']:+.3f})",
                flush=True,
            )

    macro = {}
    for calibration_size in args.calibration_sizes:
        selected = [row for row in rows if row["calibration_size"] == calibration_size]
        macro[str(calibration_size)] = {
            "subjects": len(selected),
            "base_macro_mean_deg": float(np.mean([row["base_mean_deg"] for row in selected])),
            "personalized_macro_mean_deg": float(np.mean([
                row["personalized_mean_deg"] for row in selected
            ])),
            "mean_improvement_deg": float(np.mean([
                row["improvement_mean_deg"] for row in selected
            ])),
            "subject_win_rate": float(np.mean([
                row["improvement_mean_deg"] > 0 for row in selected
            ])),
        }

    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["checkpoint"] = str(args.checkpoint)
    config["output_dir"] = str(output_dir)
    config["device"] = str(device)
    config["checkpoint_sha256"] = checkpoint_hash
    config["universal_train_sids"] = train_sids
    config["universal_val_sids"] = val_sids
    config["held_out_test_sids"] = test_sids

    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump({"config": config, "macro": macro, "subjects": rows}, handle, indent=2)
    with (output_dir / "adapters.json").open("w") as handle:
        json.dump(adapter_payload, handle, indent=2)
    with (output_dir / "split_manifest.json").open("w") as handle:
        json.dump(split_manifest, handle, indent=2)
    print(json.dumps({"output_dir": str(output_dir), "macro": macro}, indent=2), flush=True)


if __name__ == "__main__":
    main()
