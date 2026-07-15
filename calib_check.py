#!/usr/bin/env python3
"""Tuned-detector calibration on real data: does the fit stop railing?"""
import os, glob, numpy as np
from personalized_main_sequence import SaccadeDetector, MainSequenceCalibrator, PARAM_BOUNDS
D = "/home/luxliang/datasets/EXPORT_PUPIL_ALL"
def load_gaze(sid):
    rows=[]
    with open(os.path.join(D,f"{sid}.mp4gaze_vec.txt")) as f:
        for line in f.readlines()[1:]:
            q=line.strip().split(';')
            if len(q)>=4:
                try: rows.append([float(q[1]),float(q[2]),float(q[3])])
                except: pass
    return np.array(rows,np.float64)
sids = sorted(f.split('/')[-1].replace('.mp4gaze_vec.txt','')
              for f in glob.glob(os.path.join(D,"*mp4gaze_vec.txt")))
det = SaccadeDetector(fps=60.0, velocity_threshold=50.0, min_duration_ms=10.0,
                      min_amplitude_deg=0.5, max_peak_velocity=900.0)
alls=[]
for sid in sids[24:32]:
    g=load_gaze(sid); s=det.detect(g); alls+=s
    if s:
        amps=[x['amplitude'] for x in s]; vps=[x['peak_velocity'] for x in s]
        print(f"  {sid[:20]:22s} sacc={len(s):4d} amp_med={np.median(amps):5.1f} vp_med={np.median(vps):6.1f} vp_max={np.max(vps):6.1f}")
print(f"\n  pooled saccades={len(alls)}")
p=MainSequenceCalibrator.fit_all(alls)
def rail(v,key):
    lo,hi=PARAM_BOUNDS[key]; return "RAIL" if (abs(v-lo)<1e-3 or abs(v-hi)<1e-3) else "ok"
print(f"  fit: a={p['a_i']:.2f}[{rail(p['a_i'],'a_i')}] k={p['k_i']:.2f}[{rail(p['k_i'],'k_i')}] "
      f"V0={p.get('V0_i',0):.1f}[{rail(p.get('V0_i',0),'V0_i')}] tau={p.get('tau_i',0):.2f}[{rail(p.get('tau_i',0),'tau_i')}] "
      f"R2={p.get('r_squared',0):.3f}")
