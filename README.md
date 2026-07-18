# Gaze Personalization

Personalization for appearance-based 3D gaze estimation. The supported TEyeD
workflow starts from a frozen, subject-independent ResNet18-GRU checkpoint and
fits a two-parameter yaw/pitch bias for each previously unseen subject.

## Latest TEyeD result

The current result uses the universal model's six held-out subjects. For each
subject, 50 calibration-pool clips are reserved and 750 non-overlapping clips
are evaluated. Calibration and evaluation never share a segment or source
frame, and a calibration-validation gate falls back to the universal prediction
when an adapter has insufficient evidence of improvement.

| Calibration clips (K) | Universal macro mean | Personalized macro mean | Improvement | Subject wins |
|---:|---:|---:|---:|---:|
| 5 | 1.2076 deg | 1.1800 deg | 0.0276 deg | 2/6 |
| 10 | 1.2076 deg | 1.1916 deg | 0.0160 deg | 1/6 |
| **20** | **1.2076 deg** | **1.1543 deg** | **0.0533 deg (4.4%)** | **3/6** |
| 50 | 1.2076 deg | 1.1618 deg | 0.0458 deg | 3/6 |

`K=20` is the current recommended operating point. These results use the
`interleaved` session-wide protocol: calibration segments are distributed over
the recording and their neighboring segments are excluded from evaluation.
They demonstrate within-session personalization, not future-session or
strictly later-time generalization.

The harder `chronological` pilot uses only early segments for calibration. At
`K=50`, it changed the macro mean from 1.2076 deg to 1.2121 deg (-0.0045 deg),
so no positive temporal-generalization claim is made. See `FINDINGS.md` for the
full audit and negative-result analysis.

A later repeated-adapter audit found that the negative result is specific to
the two-parameter bias. A six-parameter near-identity tangent affine adapter
becomes stable from chronological `K>=15`. The current best point is `K=40`:
over 100 stratified repeats it improved the full later-time evaluation from
1.2176 deg to 1.1504 deg (mean gain 0.0672 deg, repeat standard deviation
0.0033 deg, 5th-percentile gain 0.0619 deg). This remains provisional until the
running 5-fold, 56-subject evaluation completes. The gain is checkpoint-dependent: the stronger
no-constraint universal checkpoint starts at 0.9180 deg and does not benefit
from chronological affine calibration, so absolute personalized error and the
unadapted universal baseline must always be reported together.

## Recommended workflow

`personalize_from_universal.py` enforces the evaluation boundary explicitly:

1. Reproduce the universal model's subject-disjoint split (`seed=42`).
2. Load the universal checkpoint with `strict=True` and freeze every parameter.
3. Reserve whole segments from each held-out subject as labeled calibration.
4. Exclude neighboring embargo segments before evaluating the remaining clips.
5. Report baseline and personalized metrics for every subject and calibration size.

Calibration and evaluation never share a segment or source frame. The default
TEyeD indexing settings are `segment_min_len=120`, at most 60 segments per
subject, and 15 uniformly spaced clips per segment. On the local 56-subject
export these settings reproduce the historical validation/test counts of
4,500/5,400 clips.

```bash
python3 personalize_from_universal.py \
  --data-dir /path/to/EXPORT_PUPIL_ALL \
  --checkpoint /path/to/resnet18_gru_bio_two_stage_best.pt \
  --device cuda \
  --calibration-sizes 5,10,20,50 \
  --split-strategy interleaved
```

The default `--split-strategy chronological` uses early calibration segments
and evaluates strictly later in the recording. This is the harder temporal
generalization protocol. `--split-strategy interleaved` distributes calibration
segments across the session and evaluates on non-neighboring segments; it models
a short session-wide calibration while preserving segment and frame isolation.

Outputs are isolated under `runs/personalization-<timestamp>/`:

- `results.csv`: per-subject baseline and personalized metrics
- `summary.json`: configuration, checkpoint SHA-256, and macro-subject metrics
- `adapters.json`: learned yaw/pitch biases
- `split_manifest.json`: exact calibration/evaluation frame provenance

Use `--index-only` to validate data and temporal splits without loading a model.
The evaluated universal checkpoint has SHA-256
`be2c9951c02543f262d3413c9e170b303592e47922e607af5444893ca8376564`.

## Large-scale adapter benchmark

`personalization_benchmark.py` caches the frozen universal prediction for every
indexed held-out clip, then compares three small output adapters without
re-decoding video:

- `bias`: two-parameter yaw/pitch offset
- `rotation`: three-parameter SO(3) rotation
- `affine`: six-parameter tangent-plane affine transform

It supports repeated stratified calibration sampling, both temporal protocols,
calibration-validation fallback, repeat-level uncertainty, and per-subject
stability summaries.

```bash
python3 personalization_benchmark.py \
  --data-dir /path/to/EXPORT_PUPIL_ALL \
  --checkpoint /path/to/resnet18_gru_bio_two_stage_best.pt \
  --cache-dir /large_disk/gaze-personalization-cache \
  --output-dir /large_disk/gaze-personalization-benchmark \
  --device cuda:0 \
  --methods bias,rotation,affine \
  --protocols chronological,interleaved \
  --calibration-sizes 5,10,20,50 \
  --repeats 20
```

The output contains `results.csv`, `summary.csv`, `subject_summary.csv`,
`summary.json`, and an exact `split_manifest.json`.

For full subject-disjoint cross-validation, generate folds with
`make_subject_cv_folds.py`. `train_ours_two_stage.py --split-json ...
--fold-index ...` then trains a universal checkpoint on an explicit fold and
writes the exact subject split next to the checkpoint.

## Files

- `personalize_from_universal.py` - frozen-universal calibration and evaluation
- `personalization_benchmark.py` - cached repeated bias/SO(3)/affine benchmark
- `make_subject_cv_folds.py` - deterministic subject-disjoint CV fold generator
- `train_ours_two_stage.py` - population training and experimental subject conditioning
- `personalized_main_sequence.py` - Main Sequence detector/calibrator/verifier research code
- `ablation_teyed_with_ms.py` - historical TEyeD Main Sequence ablation
- `tests/` - unit and regression tests
- `FINDINGS.md` - experiment findings and known limitations

The `feat_scale` and `hidden_init` modes in `train_ours_two_stage.py` condition
training subjects. They are not, by themselves, an unseen-user calibration
protocol. Their projections now initialize to the universal identity, inference
accepts `subject_idx`, checkpoints are separated by mode, and EMA-selected
weights are saved consistently.

## Main Sequence limitation

Personalized Main Sequence constraints require data fast enough to resolve
20-80 ms saccades. On this 60 fps export, offline fits have near-zero R-squared,
the historical training path has no usable saccade dynamics, and inference
verification did not improve angular error. Treat this branch as research code
for future high-speed (preferably at least 250 fps) data, not as the official
TEyeD personalization path.

## Tests

```bash
python3 -m pytest -q
```

The current suite contains 30 passing tests, including regression checks for
adapter identity, SO(3)/affine recovery, guarded fallback, subject-disjoint CV,
subject/layer hidden-state layout, and both split strategies' segment/frame
embargoes.

## Authentication

`upload_and_run.py` uses the local SSH agent/key through `scp` and `ssh`. It does
not store server passwords or API tokens. Configure paths through CLI arguments
or environment variables (`GAZE_HOST`, `GAZE_USER`, `EYE_DATA_DIR`, and
`UNIVERSAL_CHECKPOINT`).
