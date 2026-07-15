# Gaze Personalization

Personalized main sequence constraints for appearance-based gaze estimation.

## Files

- `personalized_main_sequence.py` — Core module: SaccadeDetector, MainSequenceCalibrator, RLS/Bayesian online adaptation, UserMainSequenceBank, MainSequenceConstraintLoss, MainSequenceVerifier
- `test_personalized_ms.py` — Simulated end-to-end demo with 3 synthetic users (Fast/Normal/Slow)
- `ablation_teyed_with_ms.py` — TEyeD dataset ablation: with vs without personalized MS constraints
- `tests/test_core.py` — pytest suite (15 cases) with regression guards
- `FINDINGS.md` — iteration/optimization findings, fixes, and the key negative result

## Requirements

```
torch>=1.9.0
numpy
scipy
torchvision
opencv-python
tqdm
pytest        # 仅测试
decord        # 仅 ablation 视频解码（否则回退 torchvision.read_video）
```

## Run

```bash
# 单元测试
python -m pytest tests/ -q

# 核心模块自测演示
python personalized_main_sequence.py

# ablation（DATA_DIR 可用环境变量覆盖，默认 TEyeD 导出目录）
GAZE_DATA_DIR=/path/to/teyed GAZE_EPOCHS=15 python ablation_teyed_with_ms.py compare
#   可选: GAZE_MAX_VIDEOS(烟囱测试) GAZE_BATCH GAZE_DEVICE
```

## 已修复（见 FINDINGS.md）

- SaccadeDetector 末尾眼跳丢失 + 生理峰速上限拒绝眨眼伪影
- RLS 在线更新截距发散 → 一致 EW-RLS（默认 λ=1.0，收敛到 OLS，与 Bayesian 一致）
- 15 用例 pytest 套件锁定回归

## 重要局限

个性化 Main Sequence 约束需要**能分辨 saccade 动态的高帧率数据（≥250fps）**。
在 60fps 数据上：saccade 时长量化过粗（duration–amplitude 拟合 R²≈0），训练时的
时序 MS 约束因 800ms 稀采样而不可见，推理时验证对角误差无正收益（Δ≈−0.02°）。
本库的机理组件已正确且有测试覆盖，适用于高帧率场景。详见 `FINDINGS.md`。
