#!/usr/bin/env python3
"""Diagnose why SaccadeDetector under-detects on real 60fps gaze.
Look at velocity distribution and saccade counts vs (v_thresh, min_dur)."""
import os, glob, numpy as np
from personalized_main_sequence import SaccadeDetector

D = "/home/luxliang/datasets/EXPORT_PUPIL_ALL"
def load_gaze(sid):
    rows=[]
    with open(os.path.join(D, f"{sid}.mp4gaze_vec.txt")) as f:
        for line in f.readlines()[1:]:
            q=line.strip().split(';')
            if len(q)>=4:
                try: rows.append([float(q[1]),float(q[2]),float(q[3])])
                except: pass
    return np.array(rows,np.float64)

sids = sorted(f.split('/')[-1].replace('.mp4gaze_vec.txt','')
              for f in glob.glob(os.path.join(D,"*mp4gaze_vec.txt")))
for sid in sids[26:29]:            # a few test-region videos
    g = load_gaze(sid)
    d = SaccadeDetector(fps=60.0)
    vel = d.angular_velocity(g)
    print(f"\n{sid} frames={len(g)}")
    print(f"  vel percentiles °/s: 50={np.percentile(vel,50):.1f} 90={np.percentile(vel,90):.1f} "
          f"99={np.percentile(vel,99):.1f} 99.9={np.percentile(vel,99.9):.1f} max={vel.max():.1f}")
    print(f"  frames>50°/s: {np.mean(vel>50)*100:.2f}%   >100°/s: {np.mean(vel>100)*100:.3f}%")
    for vt in (30, 50, 80):
        for md in (10, 20):
            dd = SaccadeDetector(fps=60.0, velocity_threshold=vt, min_duration_ms=md, min_amplitude_deg=0.5)
            n = len(dd.detect(g))
            print(f"    v_thresh={vt:3d} min_dur={md:2d}ms min_amp=0.5 -> {n:4d} saccades")
