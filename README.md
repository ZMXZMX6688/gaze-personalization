# Gaze Personalization

Personalized main sequence constraints for appearance-based gaze estimation.

## Files

- `personalized_main_sequence.py` — Core module: SaccadeDetector, MainSequenceCalibrator, RLS/Bayesian online adaptation, UserMainSequenceBank, MainSequenceConstraintLoss, MainSequenceVerifier
- `test_personalized_ms.py` — Simulated end-to-end test with 3 synthetic users (Fast/Normal/Slow)
- `ablation_teyed_with_ms.py` — TEyeD dataset ablation: with vs without personalized MS constraints

## Requirements

```
torch>=1.9.0
numpy
scipy
torchvision
opencv-python
tqdm
```
