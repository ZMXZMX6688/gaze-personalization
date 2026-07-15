#!/usr/bin/env python3
"""
Error-tail analysis: is the mean angular error inflated by GT blink/tracking-loss
frames (marked invalid in *validity_pupil.txt)? And can a runtime-available signal
(no GT) flag those frames?
"""
import os, glob, numpy as np, torch, torch.nn.functional as F
from decord import VideoReader, cpu
from ablation_teyed_with_ms import ResNet18GRUBio

D = "/home/luxliang/datasets/EXPORT_PUPIL_ALL"
CKPT = os.path.join(D, "resnet18_gru_bio_final_best.pt")
IMG, CLIP, STRIDE = 240, 8, 32
device = torch.device("cuda:0")
W0, WLEN, BATCH = 8000, 4000, 128
SIDS = os.environ.get("GAZE_SIDS", "NVIDIAAR_36_1,NVIDIAAR_39_1").split(",")

def load_col(path, col, cast):
    out=[]
    with open(path) as f:
        for line in f.readlines()[1:]:
            q=line.strip().split(';')
            if len(q)>col:
                try: out.append(cast(q[col]))
                except: out.append(cast(0))
    return np.array(out)

def load_gaze(sid):
    rows=[]
    with open(os.path.join(D,f"{sid}.mp4gaze_vec.txt")) as f:
        for line in f.readlines()[1:]:
            q=line.strip().split(';')
            if len(q)>=4:
                try: rows.append([float(q[1]),float(q[2]),float(q[3])])
                except: pass
    return np.array(rows,np.float32)

def ang_err(pred,tgt):
    p=pred/(np.linalg.norm(pred,axis=-1,keepdims=True)+1e-8)
    t=tgt/(np.linalg.norm(tgt,axis=-1,keepdims=True)+1e-8)
    return np.degrees(np.arccos(np.clip(np.sum(p*t,-1),-0.9999,0.9999)))

def dense_predict(model, vr, gaze, w0, wlen):
    lookback=(CLIP-1)*STRIDE; w0=max(w0,lookback)
    nf=min(len(vr),len(gaze)); wlen=min(wlen,nf-w0-1)
    lo=w0-lookback; hi=w0+wlen
    fr=torch.from_numpy(vr.get_batch(list(range(lo,hi))).asnumpy()).permute(0,3,1,2).float()/255.0
    if fr.shape[2]!=IMG: fr=F.interpolate(fr,size=IMG,mode='bilinear',align_corners=False)
    preds=np.zeros((wlen,3),np.float32); tg=list(range(w0,w0+wlen))
    with torch.no_grad():
        for b0 in range(0,wlen,BATCH):
            bt=tg[b0:b0+BATCH]
            x=torch.stack([fr[[t-(CLIP-1-i)*STRIDE-lo for i in range(CLIP)]] for t in bt]).to(device)
            p,_=model(x); preds[b0:b0+len(bt)]=F.normalize(p,dim=-1).cpu().numpy()
    return preds, np.arange(w0,w0+wlen)

model=ResNet18GRUBio().to(device).eval()
model.load_state_dict(torch.load(CKPT,map_location="cpu",weights_only=True))

allE=[]; allV=[]; allS=[]
for sid in SIDS:
    gaze=load_gaze(sid)
    vpath=os.path.join(D,f"{sid}.mp4pupil_seg_3D.mp4")
    valp=os.path.join(D,f"{sid}.mp4validity_pupil.txt")
    if not os.path.exists(valp):
        print(f"  [skip] {sid}: no validity file"); continue
    validity=load_col(valp,1,int)
    vr=VideoReader(vpath,ctx=cpu(0))
    preds,idx=dense_predict(model,vr,gaze,W0,WLEN)
    e=ang_err(preds,gaze[idx])
    v=(validity[idx]==1)
    # runtime signal (no GT): prediction temporal jump (deg between consecutive preds)
    jump=np.zeros(len(preds));
    d=np.degrees(np.arccos(np.clip(np.sum(preds[:-1]*preds[1:],-1),-1,1))); jump[1:]=d
    allE.append(e); allV.append(v); allS.append(jump)
    print(f"\n{sid}: {len(e)} frames, invalid={np.mean(~v)*100:.1f}%")
    print(f"  raw          mean={e.mean():6.3f}° median={np.median(e):.3f}° p95={np.percentile(e,95):.2f}° max={e.max():.1f}°")
    if v.sum()>5:
        ev=e[v]
        print(f"  VALID only   mean={ev.mean():6.3f}° median={np.median(ev):.3f}° p95={np.percentile(ev,95):.2f}° max={ev.max():.1f}°")
    if (~v).sum()>0:
        print(f"  invalid only mean={e[~v].mean():6.3f}°  (blink/tracking-loss GT)")
    # how much of the error tail is invalid frames?
    thr=np.percentile(e,95); tail=e>=thr
    print(f"  top-5% error frames: {np.mean(~v[tail])*100:.1f}% are GT-invalid")

if allE:
    e=np.concatenate(allE); v=np.concatenate(allV); s=np.concatenate(allS)
    print("\n=== AGGREGATE ===")
    print(f"  raw        mean={e.mean():.3f}°   (n={len(e)}, invalid={np.mean(~v)*100:.1f}%)")
    print(f"  VALID only mean={e[v].mean():.3f}°   <- honest model error")
    print(f"  Δ from dropping bad-label frames: {e.mean()-e[v].mean():+.3f}°")
    # runtime rejection: flag frames by prediction jump, no GT needed
    for q in (90,95,99):
        thr=np.percentile(s,q); keep=s<thr
        print(f"  runtime reject top-{100-q}% by pred-jump: kept mean={e[keep].mean():.3f}° "
              f"(recall of invalid frames flagged={np.mean(s[~v]>=thr)*100:.0f}%)")
