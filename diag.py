#!/usr/bin/env python3
"""Diagnostic harness: trace RLS divergence and SaccadeDetector behavior."""
import numpy as np
from personalized_main_sequence import RLSMainSequence, BayesianMainSequence, SaccadeDetector

np.random.seed(0)
TRUE_A, TRUE_K = 2.0, 2.5

print("="*70)
print("RLS config sweep — final |Da|+|Dk| averaged over 200 runs (true a=2.0 k=2.5)")
print("="*70)
configs = [
    dict(P_init=1.0,  obs_noise=1.5, lambda_forget=1.0),
    dict(P_init=1.0,  obs_noise=3.0, lambda_forget=1.0),
    dict(P_init=1.0,  obs_noise=5.0, lambda_forget=1.0),
    dict(P_init=0.5,  obs_noise=5.0, lambda_forget=1.0),
    dict(P_init=10.0, obs_noise=1.5, lambda_forget=1.0),
    dict(P_init=1.0,  obs_noise=5.0, lambda_forget=0.995),
]
def run_rls(cfg, seed):
    rng = np.random.RandomState(seed)
    r = RLSMainSequence(**cfg)
    for _ in range(50):
        amp = rng.uniform(2, 20)
        dur = TRUE_A + TRUE_K*amp + rng.randn()*1.5
        out = r.update(amp, dur)
    return abs(out['a_i']-TRUE_A), abs(out['k_i']-TRUE_K)
def run_bay(seed):
    rng = np.random.RandomState(seed)
    b = BayesianMainSequence()
    for _ in range(50):
        amp = rng.uniform(2, 20)
        dur = TRUE_A + TRUE_K*amp + rng.randn()*1.5
        out = b.update(amp, dur)
    return abs(out['a_i']-TRUE_A), abs(out['k_i']-TRUE_K)
for cfg in configs:
    errs = np.array([run_rls(cfg, s) for s in range(200)])
    da, dk = errs.mean(0)
    print(f"  P0={cfg['P_init']:4.1f} R_std={cfg['obs_noise']:3.1f} lam={cfg['lambda_forget']:.3f}"
          f"  ->  <|Da|>={da:.3f}  <|Dk|>={dk:.3f}  <sum>={da+dk:.3f}")
berrs = np.array([run_bay(s) for s in range(200)])
bda, bdk = berrs.mean(0)
print(f"  Bayesian (reference)                        ->  <|Da|>={bda:.3f}  <|Dk|>={bdk:.3f}  <sum>={bda+bdk:.3f}")

print("\n"+"="*70)
print("SaccadeDetector — clean synthetic (return-to-center between saccades)")
print("="*70)
fps = 60.0
det = SaccadeDetector(fps=fps)
T = 400
gaze = np.tile([0.,0.,1.], (T,1)).astype(np.float64)
truth = []
# saccades: (start_frame, amp_deg, n_frames_motion); return to center after each
pos = 0
f = 20
for amp, nf in [(10,6),(15,8),(8,5),(20,7),(5,4)]:
    end = np.array([np.sin(np.radians(amp)),0,np.cos(np.radians(amp))])
    start = gaze[f-1].copy()
    for t in range(nf):
        a = t/(nf-1)
        v = (1-a)*start + a*end
        gaze[f+t] = v/np.linalg.norm(v)
    # hold for 25 frames
    for t in range(nf, nf+25):
        if f+t < T: gaze[f+t] = end
    truth.append((f, amp, nf))
    # return to center
    f2 = f+nf+25
    startb = end
    for t in range(6):
        if f2+t>=T: break
        a=t/5
        v=(1-a)*startb + a*np.array([0.,0.,1.])
        gaze[f2+t]=v/np.linalg.norm(v)
    for t in range(6, 30):
        if f2+t<T: gaze[f2+t]=np.array([0.,0.,1.])
    f = f2+30
    if f>=T-40: break

vel = det.angular_velocity(gaze)
print(f"  velocity: max={vel.max():.1f} deg/s, n>50={np.sum(vel>50)}")
detected = det.detect(gaze)
print(f"  injected {len(truth)} saccades (forward only, amps={[a for _,a,_ in truth]})")
print(f"  detected {len(detected)}:")
for d in detected:
    print(f"    A={d['amplitude']:5.1f} D={d['duration']:6.1f}ms Vp={d['peak_velocity']:6.1f} "
          f"frames[{d['start_frame']}-{d['end_frame']}]")
