#!/usr/bin/env python3
"""
推理时 Main Sequence 验证实验。
用现有 checkpoint 在密集 60fps 序列上做逐帧预测（内部clip配置=训练配置），
然后用 per-user 离线标定的 MS 参数 + MainSequenceVerifier 纠正，对比角误差。

env: GAZE_WLEN(窗口帧数) GAZE_NVID(测试视频数) GAZE_W0(窗口起点)
"""
import os, glob, numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
from ablation_teyed_with_ms import ResNet18GRUBio
from personalized_main_sequence import (
    SaccadeDetector, MainSequenceCalibrator, MainSequenceVerifier, POPULATION_PRIOR)

D = "/home/luxliang/datasets/EXPORT_PUPIL_ALL"
CKPT = os.path.join(D, "resnet18_gru_bio_final_best.pt")
IMG, CLIP, STRIDE = 240, 8, 32          # identified training config
FPS = 60.0
device = torch.device("cuda:0")
WLEN = int(os.environ.get("GAZE_WLEN", 1500))
NVID = int(os.environ.get("GAZE_NVID", 2))
W0   = int(os.environ.get("GAZE_W0", 5000))
BATCH = 128

def load_gaze(sid):
    rows = []
    with open(os.path.join(D, f"{sid}.mp4gaze_vec.txt")) as f:
        for line in f.readlines()[1:]:
            q = line.strip().split(';')
            if len(q) >= 4:
                try: rows.append([float(q[1]), float(q[2]), float(q[3])])
                except: pass
    return np.array(rows, np.float32)

def ang_err_np(pred, tgt):
    p = pred / (np.linalg.norm(pred, axis=-1, keepdims=True)+1e-8)
    t = tgt  / (np.linalg.norm(tgt, axis=-1, keepdims=True)+1e-8)
    c = np.clip(np.sum(p*t, -1), -0.9999, 0.9999)
    return np.degrees(np.arccos(c))

def dense_predict(model, vr, gaze, w0, wlen):
    """逐帧(stride-1)滑窗预测：target t 的 clip=[t-(CLIP-1)*STRIDE, ..., t]"""
    lookback = (CLIP-1)*STRIDE
    w0 = max(w0, lookback)
    nframe = min(len(vr), len(gaze))
    wlen = min(wlen, nframe - w0 - 1)
    lo = w0 - lookback
    hi = w0 + wlen
    frames = vr.get_batch(list(range(lo, hi))).asnumpy()      # [F,H,W,C] uint8
    frames = torch.from_numpy(frames).permute(0,3,1,2).float()/255.0
    if frames.shape[2]!=IMG or frames.shape[3]!=IMG:
        frames = F.interpolate(frames, size=IMG, mode='bilinear', align_corners=False)
    preds = np.zeros((wlen,3), np.float32)
    targets = list(range(w0, w0+wlen))
    with torch.no_grad():
        for b0 in range(0, wlen, BATCH):
            bt = targets[b0:b0+BATCH]
            clips = []
            for t in bt:
                idx = [t - (CLIP-1-i)*STRIDE - lo for i in range(CLIP)]
                clips.append(frames[idx])
            x = torch.stack(clips).to(device)                 # [B,CLIP,C,H,W]
            p,_ = model(x)
            preds[b0:b0+len(bt)] = F.normalize(p,dim=-1).cpu().numpy()
    gt = gaze[w0:w0+wlen]
    return preds, gt

def summarize(tag, e):
    print(f"    {tag:16s} n={len(e):5d} mean={e.mean():6.3f}° median={np.median(e):6.3f}° "
          f"p90={np.percentile(e,90):6.3f}° max={e.max():6.2f}°")

def gt_saccade_mask(gt, fps=FPS, vthr=50.0, dilate=3):
    """GT速度>阈值的帧(±dilate膨胀)：验证器应起作用的区域。"""
    g = gt/(np.linalg.norm(gt,axis=1,keepdims=True)+1e-8)
    cs = np.clip(np.sum(g[:-1]*g[1:],1),-1,1)
    vel = np.degrees(np.arccos(cs))*fps
    m = np.zeros(len(gt), bool)
    hi = np.where(vel>vthr)[0]
    for i in hi:
        m[max(0,i-dilate):min(len(gt),i+dilate+1)] = True
    return m

def main():
    model = ResNet18GRUBio().to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    det = SaccadeDetector(fps=FPS, velocity_threshold=50.0, min_duration_ms=10.0,
                          min_amplitude_deg=0.5, max_peak_velocity=900.0)
    sids = sorted(f.split('/')[-1].replace('.mp4gaze_vec.txt','')
                  for f in glob.glob(os.path.join(D, "*mp4gaze_vec.txt")))
    env_sids = os.environ.get("GAZE_SIDS", "").strip()
    if env_sids:
        test_sids = env_sids.split(",")[:NVID]
    else:
        test_sids = sids[int(len(sids)*0.85):][:NVID]     # 与训练脚本一致的test划分
    all_base, all_corr, all_corr_pop = [], [], []
    for sid in test_sids:
        vpath = os.path.join(D, f"{sid}.mp4pupil_seg_3D.mp4")
        gaze = load_gaze(sid)
        if len(gaze) < W0 + WLEN + 10:
            print(f"  [skip] {sid} too short ({len(gaze)})"); continue
        vr = VideoReader(vpath, ctx=cpu(0))
        preds, gt = dense_predict(model, vr, gaze, W0, WLEN)
        base_e = ang_err_np(preds, gt)

        # per-user 标定（用整段dense GT）
        saccades = det.detect(gaze)
        params = MainSequenceCalibrator.fit_all(saccades) if len(saccades)>=4 else POPULATION_PRIOR.copy()

        def verify(pms):
            v = MainSequenceVerifier(personalized_params=pms, fps=FPS)
            out = np.zeros_like(preds)
            for i in range(len(preds)):
                out[i] = v.step(preds[i].copy())['corrected']
            return ang_err_np(out, gt)
        corr_e = verify(params)
        corr_pop_e = verify(POPULATION_PRIOR.copy())

        mask = gt_saccade_mask(gt)
        print(f"  {sid}  frames={len(gaze)}  saccades={len(saccades)}  "
              f"fit(a={params.get('a_i',0):.2f},k={params.get('k_i',0):.2f},R2={params.get('r_squared',0):.3f})  "
              f"GT-saccade-frames={mask.sum()}/{len(gt)}")
        print("   [ALL frames]")
        summarize("baseline", base_e); summarize("verify(user)", corr_e); summarize("verify(pop)", corr_pop_e)
        if mask.sum() > 5:
            print("   [GT-saccade frames only]")
            summarize("baseline", base_e[mask]); summarize("verify(user)", corr_e[mask]); summarize("verify(pop)", corr_pop_e[mask])
        all_base.append(base_e); all_corr.append(corr_e); all_corr_pop.append(corr_pop_e)

    if all_base:
        b=np.concatenate(all_base); c=np.concatenate(all_corr); cp=np.concatenate(all_corr_pop)
        print("\n  === AGGREGATE (all frames) ===")
        summarize("baseline", b); summarize("verify(user)", c); summarize("verify(pop)", cp)
        print(f"    Δmean(user)={b.mean()-c.mean():+.4f}°  Δmean(pop)={b.mean()-cp.mean():+.4f}°  (正=改善)")

if __name__ == "__main__":
    main()
