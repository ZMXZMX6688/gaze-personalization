#!/usr/bin/env python3
"""Probe: does SaccadeDetector find real saccades in the dense 60fps gaze_vec data?
If yes, the offline-calibration + inference-verification path is viable."""
import numpy as np, glob, os
from personalized_main_sequence import SaccadeDetector, MainSequenceCalibrator

D = "/home/luxliang/datasets/EXPORT_PUPIL_ALL"
files = sorted(glob.glob(os.path.join(D, "*mp4gaze_vec.txt")))[:8]
det = SaccadeDetector(fps=60.0)

def load(p):
    rows = []
    with open(p) as f:
        for line in f.readlines()[1:]:
            q = line.strip().split(';')
            if len(q) >= 4:
                try: rows.append([float(q[1]), float(q[2]), float(q[3])])
                except: pass
    return np.array(rows, dtype=np.float64)

total = []
for p in files:
    g = load(p)
    if len(g) < 10:
        continue
    sacc = det.detect(g)
    total += sacc
    amps = [s['amplitude'] for s in sacc]
    print(f"  {os.path.basename(p)[:22]:24s} frames={len(g):5d} saccades={len(sacc):4d} "
          f"amp[min/med/max]={np.min(amps):.1f}/{np.median(amps):.1f}/{np.max(amps):.1f}" if sacc
          else f"  {os.path.basename(p)[:22]:24s} frames={len(g):5d} saccades=0")

print(f"\n  TOTAL saccades across {len(files)} videos: {len(total)}")
if len(total) >= 4:
    p = MainSequenceCalibrator.fit_all(total)
    print(f"  Fitted population main-sequence from REAL data:")
    print(f"    a={p['a_i']:.3f} ms, k={p['k_i']:.3f} ms/deg, "
          f"V0={p.get('V0_i',0):.1f} deg/s, tau={p.get('tau_i',0):.2f} deg, "
          f"R2={p.get('r_squared',0):.3f}")
