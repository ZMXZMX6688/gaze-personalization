#!/usr/bin/env python3
"""Identify the checkpoint's training clip-config by which one yields low angular error.
Also confirms strict state_dict load (architecture match)."""
import os, glob, numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
from ablation_teyed_with_ms import ResNet18GRUBio

D = "/home/luxliang/datasets/EXPORT_PUPIL_ALL"
CKPT = os.path.join(D, "resnet18_gru_bio_final_best.pt")
device = torch.device("cuda:0")

def load_gaze(sid):
    rows = []
    with open(os.path.join(D, f"{sid}.mp4gaze_vec.txt")) as f:
        for line in f.readlines()[1:]:
            q = line.strip().split(';')
            if len(q) >= 4:
                try: rows.append([float(q[1]), float(q[2]), float(q[3])])
                except: pass
    return np.array(rows, np.float32)

def ang_err(pred, tgt):
    p = F.normalize(pred, dim=-1); t = F.normalize(tgt, dim=-1)
    c = F.cosine_similarity(p, t, dim=-1).clamp(-0.9999, 0.9999)
    return (torch.acos(c) * 180 / np.pi)

# pick a mid video with enough frames
sids = sorted(f.split('/')[-1].replace('.mp4gaze_vec.txt','')
              for f in glob.glob(os.path.join(D, "*mp4gaze_vec.txt")))
sid = sids[len(sids)//2]
vpath = os.path.join(D, f"{sid}.mp4pupil_seg_3D.mp4")
gaze = load_gaze(sid)
vr = VideoReader(vpath, ctx=cpu(0))
nframe = min(len(vr), len(gaze))
print(f"probe video={sid} frames={nframe}")

model = ResNet18GRUBio().to(device).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"strict-load: missing={len(missing)} unexpected={len(unexpected)}"
      + (f" e.g. {missing[:2]}{unexpected[:2]}" if (missing or unexpected) else " (exact match)"))

configs = [
    dict(name="224/4/48", img=224, clip=4, stride=48),
    dict(name="240/8/32", img=240, clip=8, stride=32),
    dict(name="240/4/48", img=240, clip=4, stride=48),
    dict(name="224/8/32", img=224, clip=8, stride=32),
]
for cfg in configs:
    need = (cfg['clip']-1)*cfg['stride'] + 1
    starts = list(range(0, nframe - need, max(cfg['stride'], (nframe-need)//120)))[:120]
    errs = []
    with torch.no_grad():
        for s in starts:
            idx = [s + i*cfg['stride'] for i in range(cfg['clip'])]
            fr = vr.get_batch(idx).asnumpy().astype(np.float32)/255.0   # [T,H,W,C]
            x = torch.from_numpy(fr).permute(0,3,1,2)
            if x.shape[2]!=cfg['img'] or x.shape[3]!=cfg['img']:
                x = F.interpolate(x, size=cfg['img'], mode='bilinear', align_corners=False)
            x = x.unsqueeze(0).to(device)  # [1,T,C,H,W]
            pred,_ = model(x)
            tgt = torch.from_numpy(gaze[idx[-1]]).unsqueeze(0).to(device)
            errs.append(ang_err(pred, tgt).item())
    errs = np.array(errs)
    print(f"  cfg {cfg['name']:9s} n={len(errs):3d}  mean={errs.mean():6.2f}°  median={np.median(errs):6.2f}°")
