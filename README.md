# Gaze Personalization

Personalization for appearance-based 3D gaze estimation. The supported TEyeD
workflow starts from a frozen, subject-independent ResNet18-GRU checkpoint and
fits a two-parameter yaw/pitch bias for each previously unseen subject.

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
python personalize_from_universal.py \
  --data-dir /path/to/EXPORT_PUPIL_ALL \
  --checkpoint /path/to/resnet18_gru_bio_two_stage_best.pt \
  --device cuda \
  --calibration-sizes 5,10,20,50
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

## Files

- `personalize_from_universal.py` - frozen-universal calibration and evaluation
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
python -m pytest -q
```

## Authentication

`upload_and_run.py` uses the local SSH agent/key through `scp` and `ssh`. It does
not store server passwords or API tokens. Configure paths through CLI arguments
or environment variables (`GAZE_HOST`, `GAZE_USER`, `EYE_DATA_DIR`, and
`UNIVERSAL_CHECKPOINT`).
