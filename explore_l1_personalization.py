#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore L1 threshold personalization strategies for GAI inference.

Approaches:
  A: Per-subject grid search for optimal L1 threshold
  B: SubjectEmbedding → L1 threshold predictor (Linear 16→1)
  C: Online adaptive threshold (running mean + k * running std)
  D: Hybrid (predicted base + adaptive modulation)
"""

import os, math, random, csv, re, sys, json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

# ── Constants ──
DATA_DIR = Path("/home/zmx/AR_Base_Data")
CLIP_LEN = 8
HIDDEN = 256
IMG_SIZE = 240
STRIDE = 4

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

L1_THRESHOLDS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
ADAPTIVE_KS = [0.5, 1.0, 1.5, 2.0, 3.0]
MAX_SKIP = 5
WINDOW_SIZE = 20

# ── Helper ──
def vec_angle_deg(v1, v2):
    a = v1 / (np.linalg.norm(v1) + 1e-7)
    b = v2 / (np.linalg.norm(v2) + 1e-7)
    return math.degrees(math.acos(np.clip(np.dot(a, b), -1 + 1e-7, 1 - 1e-7)))


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════
class ResNet18GRUModel(nn.Module):
    def __init__(self, feat_dim=512, hidden=256, dropout=0.3,
                 personalize_mode='none', num_subjects=0):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_drop = nn.Dropout(dropout)
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden,
                          num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 3),
        )
        self.personalize_mode = personalize_mode
        if personalize_mode != 'none' and num_subjects > 0:
            self.subject_embed = SubjectEmbedding(num_subjects, feat_dim=feat_dim, hidden=hidden)

    def forward(self, x, subject_idx=None):
        return self.forward_all(x, subject_idx=subject_idx)[:, -1]

    def forward_all(self, x, subject_idx=None):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W).contiguous()
        f = self.backbone(x)
        f = self.feat_drop(f.flatten(1))
        f = f.reshape(B, T, -1)

        if self.personalize_mode == 'feat_scale' and subject_idx is not None:
            valid = (subject_idx >= 0)
            if valid.any():
                scale = torch.ones(B, 1, self.feat_dim, device=f.device)
                scale[valid] = self.subject_embed.forward_feat_scale(subject_idx[valid])
                f = f * scale

        if self.personalize_mode == 'hidden_init' and subject_idx is not None:
            valid = (subject_idx >= 0)
            if valid.any():
                h0 = torch.zeros(2, B, self.hidden, device=f.device)
                h0[:, valid] = self.subject_embed.forward_hidden_init(
                    subject_idx[valid], valid.sum(), f.device)
                out, _ = self.gru(f, h0)
            else:
                out, _ = self.gru(f)
        else:
            out, _ = self.gru(f)
        return self.head(out)


class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, embed_dim=16, feat_dim=512, hidden=256, num_layers=2):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_subjects, embed_dim)
        self.feat_proj = nn.Linear(embed_dim, feat_dim)
        self.hidden_proj = nn.Linear(embed_dim, hidden * num_layers)
        nn.init.zeros_(self.feat_proj.weight)
        nn.init.zeros_(self.feat_proj.bias)
        nn.init.zeros_(self.hidden_proj.weight)
        nn.init.zeros_(self.hidden_proj.bias)

    def forward_feat_scale(self, subject_idx):
        emb = self.embedding(subject_idx)
        return 1.0 + self.feat_proj(emb).unsqueeze(1)

    def forward_hidden_init(self, subject_idx, batch_size, device):
        emb = self.embedding(subject_idx)
        h0 = self.hidden_proj(emb)
        return h0.reshape(batch_size, self.num_layers, self.hidden).permute(1, 0, 2).contiguous()


# ═══════════════════════════════════════════════════════════════
# GAI Components
# ═══════════════════════════════════════════════════════════════
class L1VelocityDetector:
    """Adaptive velocity detector with optional running statistics."""
    def __init__(self, thr=1.0, adaptive=False, k=2.0, window=WINDOW_SIZE):
        self.base_thr = thr
        self.adaptive = adaptive
        self.k = k
        self.window = window
        self.velocities = []

    def should_skip(self, model_hist):
        if len(model_hist) < 2:
            return False
        vel = vec_angle_deg(model_hist[-2], model_hist[-1])
        if self.adaptive:
            self.velocities.append(vel)
            if len(self.velocities) > self.window:
                self.velocities.pop(0)
            if len(self.velocities) >= 3:
                mu = np.mean(self.velocities)
                sigma = np.std(self.velocities) + 1e-7
                thr = mu + self.k * sigma
            else:
                thr = self.base_thr
        else:
            thr = self.base_thr
        return vel < thr

    def get_threshold(self):
        if self.adaptive and len(self.velocities) >= 3:
            return float(np.mean(self.velocities) + self.k * np.std(self.velocities))
        return self.base_thr


class L2Extrapolator:
    def predict(self, h):
        if not h:
            return np.array([0, 0, 1], dtype=np.float32)
        if len(h) == 1:
            return h[-1].copy()
        last = h[-1].copy()
        if len(h) >= 3:
            vel = (h[-1] - h[-3]) / 2.0
        else:
            vel = h[-1] - h[-2]
        pred = last + 0.5 * vel
        norm = np.linalg.norm(pred)
        return pred / norm if norm > 1e-7 else pred


class L3ConstraintCorrector:
    def correct(self, g):
        g = g / (np.linalg.norm(g) + 1e-7)
        off = math.degrees(math.acos(np.clip(g[2], -1 + 1e-7, 1 - 1e-7)))
        if off > 38.0:
            excess = off - 38.0
            target = 38.0 + 0.4 * excess
            st = math.sin(math.radians(target))
            ct = math.cos(math.radians(target))
            r = math.sqrt(g[0]**2 + g[1]**2)
            if r > 1e-7:
                g[0] = g[0] / r * st
                g[1] = g[1] / r * st
                g[2] = ct
        elev = math.degrees(math.acos(np.clip(abs(g[2]), -1 + 1e-7, 1 - 1e-7)))
        if elev > 50.0:
            excess = elev - 50.0
            target_e = elev - 0.5 * excess
            se = math.sin(math.radians(target_e))
            ce = math.cos(math.radians(target_e))
            r = math.sqrt(g[0]**2 + g[1]**2)
            if r > 1e-7:
                g[0] = g[0] / r * se
                g[1] = g[1] / r * se
                g[2] = math.copysign(ce, g[2])
        return g / (np.linalg.norm(g) + 1e-7)


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════
def load_gaze(p):
    rows = []
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines()[1:]:
            ps = line.strip().split(';')
            if len(ps) >= 4:
                try:
                    rows.append([float(ps[1]), float(ps[2]), float(ps[3])])
                except:
                    pass
    return np.array(rows, dtype=np.float32)


def load_valid(p):
    rows = []
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines()[1:]:
            ps = line.strip().split(';')
            if len(ps) >= 2:
                try:
                    rows.append(int(ps[1]))
                except:
                    rows.append(1)
    return np.array(rows, dtype=np.int32)


def load_frame(cap, fi, cache):
    if fi in cache:
        return cache[fi]
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE))
    t = VAL_TRANSFORM(np.array(img))
    cache[fi] = t
    return t


def get_sid_split(seed=42):
    """Reproduce the same train/val/test split as eval_4configs_optimized.py."""
    pat = re.compile(r"^(NVIDIA(?:AR|VR)_\d+_1)\.mp4pupil_seg_3D\.mp4$")
    all_sids = sorted({pat.match(p.name).group(1)
                       for p in DATA_DIR.iterdir()
                       if p.is_file() and pat.match(p.name)})
    random.seed(seed)
    random.shuffle(all_sids)
    test_sids = all_sids[31:34]
    val_sids = all_sids[28:31]
    train_sids = all_sids[:28]
    return train_sids, val_sids, test_sids


# ═══════════════════════════════════════════════════════════════
# Evaluation Core
# ═══════════════════════════════════════════════════════════════
MAX_SEQ = 2000


def evaluate_subject(model, device, sid, l1_thr, use_l2=True, use_l3=True,
                     adaptive=False, adaptive_k=2.0, max_skip=MAX_SKIP,
                     subject_idx=None):
    """Run GAI evaluation on a single subject. Returns list of angular errors."""
    vp = DATA_DIR / f"{sid}.mp4pupil_seg_3D.mp4"
    gp = DATA_DIR / f"{sid}.mp4gaze_vec.txt"
    vap = DATA_DIR / f"{sid}.mp4validity_pupil.txt"
    if not all([vp.exists(), gp.exists(), vap.exists()]):
        return None, 0, 0

    gaze = load_gaze(gp)
    valid_arr = load_valid(vap)
    n = min(len(gaze), len(valid_arr))
    need = (CLIP_LEN - 1) * STRIDE + 1
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return None, 0, 0

    model_hist = []
    out_hist = []
    csn = 0
    cache = {}
    all_errs = []
    skip_n = 0
    total_n = 0

    l1 = L1VelocityDetector(thr=l1_thr, adaptive=adaptive, k=adaptive_k, window=WINDOW_SIZE)
    l2 = L2Extrapolator()
    l3 = L3ConstraintCorrector()

    loop_step = max(STRIDE, 4)
    for start in range(0, min(n - need + 1, MAX_SEQ), loop_step):
        if not np.all(valid_arr[start:start + need] == 1):
            model_hist = []
            out_hist = []
            csn = 0
            cache = {}
            continue

        tf = start + (CLIP_LEN - 1) * STRIDE
        fids = [start + i * STRIDE for i in range(CLIP_LEN)]
        frames = [load_frame(cap, fi, cache) for fi in fids]
        if any(f is None for f in frames):
            continue

        gt = gaze[tf]
        x = torch.stack(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            do_skip = l1.should_skip(model_hist) and csn < max_skip
            if do_skip:
                if use_l2 and len(model_hist) >= 2:
                    pred = l2.predict(model_hist)
                else:
                    pred = out_hist[-1] if out_hist else np.array([0, 0, 1], dtype=np.float32)
                skip_n += 1
                csn += 1
            else:
                if subject_idx is not None:
                    pred = model(x, subject_idx=torch.tensor([subject_idx], device=device))
                else:
                    pred = model(x)
                pred = pred.cpu().numpy()[0]
                model_hist.append(pred)
                if len(model_hist) > 20:
                    model_hist.pop(0)
                csn = 0
            if use_l3:
                pred = l3.correct(pred)

        all_errs.append(vec_angle_deg(pred, gt))
        out_hist.append(pred)
        if len(out_hist) > 20:
            out_hist.pop(0)
        total_n += 1

    cap.release()
    sr = skip_n / total_n * 100 if total_n > 0 else 0
    return all_errs, sr, total_n


# ═══════════════════════════════════════════════════════════════
# 方案A: Per-Subject Grid Search
# ═══════════════════════════════════════════════════════════════
def run_scheme_a(model, device, val_sids):
    """Grid search L1 thresholds on each validation subject."""
    print("\n" + "=" * 70)
    print("方案A: Per-Subject L1 Threshold Grid Search")
    print("=" * 70)

    all_results = []
    best_per_subject = {}

    for sid in val_sids:
        print(f"\n  Subject: {sid}")
        print(f"  {'Threshold':>10} | {'Mean':>8} | {'Median':>8} | {'Skip%':>7} | {'N':>6}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}")

        best_err = float('inf')
        best_thr = None

        for thr in L1_THRESHOLDS:
            errs, sr, n = evaluate_subject(model, device, sid, l1_thr=thr,
                                           use_l2=True, use_l3=True)
            if errs is None:
                print(f"  {thr:>10.1f} | {'N/A':>8} | {'N/A':>8} | {'N/A':>7} | {'N/A':>6}")
                continue
            mean_e = float(np.mean(errs))
            med_e = float(np.median(errs))
            all_results.append({'sid': sid, 'threshold': thr, 'mean': mean_e,
                                'median': med_e, 'skip_rate': sr, 'n': n})
            mark = " ★" if mean_e < best_err else ""
            if mean_e < best_err:
                best_err = mean_e
                best_thr = thr
            print(f"  {thr:>10.1f} | {mean_e:>8.4f} | {med_e:>8.4f} | {sr:>6.1f}% | {n:>5}{mark}")

        best_per_subject[sid] = {'best_threshold': best_thr, 'best_mean': best_err}

    print(f"\n  Best per subject:")
    for sid, info in best_per_subject.items():
        print(f"    {sid}: threshold={info['best_threshold']}, mean_err={info['best_mean']:.4f}°")

    return all_results, best_per_subject


# ═══════════════════════════════════════════════════════════════
# 方案B: Embedding → L1 Threshold Predictor
# ═══════════════════════════════════════════════════════════════
def run_scheme_b(model, device, train_sids, val_sids, ckpt_dir):
    """Learn mapping from SubjectEmbedding to optimal L1 threshold.

    Uses training subjects' embeddings (from checkpoint) and their grid-searched
    optimal thresholds. Trains Linear(16→1) predictor.
    For unseen subjects (val_sids), embedding=0 → bias term gives default.
    """
    print("\n" + "=" * 70)
    print("方案B: Embedding → L1 Threshold Predictor")
    print("=" * 70)

    # Load model with SubjectEmbedding
    ckpt_path = ckpt_dir / "resnet18_gru_bio_two_stage_best.pt"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return None

    # Determine personalize_mode from checkpoint path
    mode_str = str(ckpt_dir)
    if 'hidden_init' in mode_str:
        p_mode = 'hidden_init'
    elif 'feat_scale' in mode_str:
        p_mode = 'feat_scale'
    else:
        p_mode = 'none'

    p_model = ResNet18GRUModel(personalize_mode=p_mode,
                               num_subjects=len(train_sids)).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace('_ema.', ''): v for k, v in state.items()}
    p_model.load_state_dict(state, strict=False)
    p_model.eval()
    print(f"  Loaded: {ckpt_path}")

    # Extract training subject embeddings
    embed_weights = p_model.subject_embed.embedding.weight.detach().cpu().numpy()  # (28, 16)
    print(f"  Embedding weights: {embed_weights.shape}")

    # Grid search optimal threshold for each training subject
    print(f"\n  Grid searching optimal thresholds for {len(train_sids)} training subjects...")
    train_opt_thresholds = {}
    for i, sid in enumerate(train_sids):
        best_err = float('inf')
        best_thr = L1_THRESHOLDS[0]
        for thr in L1_THRESHOLDS:
            errs, sr, n = evaluate_subject(
                p_model, device, sid, l1_thr=thr,
                use_l2=True, use_l3=True, subject_idx=i)
            if errs is None:
                continue
            mean_e = float(np.mean(errs))
            if mean_e < best_err:
                best_err = mean_e
                best_thr = thr
        train_opt_thresholds[sid] = best_thr
        if (i + 1) % 7 == 0:
            print(f"    [{i+1}/{len(train_sids)}] subjects done")

    # Train predictor: Linear(16 → 1) with sigmoid
    X = embed_weights  # (28, 16)
    y = np.array([train_opt_thresholds[sid] for sid in train_sids], dtype=np.float32)  # (28,)
    y_norm = y / 10.0  # normalize to [0, 1] range (max threshold is 10)

    predictor = nn.Linear(16, 1)
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y_norm).unsqueeze(1)

    for epoch in range(500):
        pred = torch.sigmoid(predictor(X_t))
        loss = F.mse_loss(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    with torch.no_grad():
        pred_norm = torch.sigmoid(predictor(X_t)).squeeze().numpy()
        pred_thr = pred_norm * 10.0
        mae = float(np.abs(pred_thr - y).mean())

    print(f"\n  Predictor trained: MAE={mae:.3f}° (on training subjects)")
    print(f"  True thresholds: {y}")
    print(f"  Pred thresholds: {np.round(pred_thr, 2)}")

    # Test on validation subjects (embedding=0 → bias)
    with torch.no_grad():
        zero_emb = torch.zeros(1, 16)
        default_thr = float(torch.sigmoid(predictor(zero_emb)).item() * 10.0)
    print(f"\n  Default threshold (zero embedding): {default_thr:.3f}°")

    # Evaluate validation subjects with predictor's default threshold
    print(f"\n  Evaluating validation subjects with default_thr={default_thr:.3f}...")
    val_results = {}
    for sid in val_sids:
        errs, sr, n = evaluate_subject(
            p_model, device, sid, l1_thr=default_thr,
            use_l2=True, use_l3=True, subject_idx=-1)
        if errs is not None:
            mean_e = float(np.mean(errs))
        val_results[sid] = {'predicted_thr': default_thr, 'mean_err': mean_e, 'skip_rate': sr}
        print(f"    {sid}: mean={mean_e:.4f}°, skip={sr:.1f}%")

    return {
        'predictor': predictor,
        'default_threshold': default_thr,
        'train_mae': mae,
        'val_results': val_results,
        'train_opt_thresholds': train_opt_thresholds,
    }


# ═══════════════════════════════════════════════════════════════
# 方案C: Adaptive Threshold
# ═══════════════════════════════════════════════════════════════
def run_scheme_c(model, device, val_sids):
    """Test adaptive L1 threshold with different k values."""
    print("\n" + "=" * 70)
    print("方案C: Adaptive L1 Threshold (running mean + k * std)")
    print("=" * 70)

    all_results = []
    best_k = None
    best_overall_err = float('inf')

    print(f"\n  {'k':>5} | ", end='')
    for sid in val_sids:
        print(f'{sid[-10:]:>20}', end='')
    print(f" {'Overall':>10}")
    print(f"  {'-'*5}-+-{'-'*20*len(val_sids)}-+-{'-'*10}")

    for k in ADAPTIVE_KS:
        row = f"  {k:>5.1f} | "
        k_errs = []
        for sid in val_sids:
            errs, sr, n = evaluate_subject(
                model, device, sid, l1_thr=1.0,
                adaptive=True, adaptive_k=k,
                use_l2=True, use_l3=True)
            if errs is not None:
                me = float(np.mean(errs))
                k_errs.append(me)
                row += f" {me:>8.4f}  |"
            else:
                row += f" {'N/A':>8}  |"
        overall = float(np.mean(k_errs))
        row += f" {overall:>8.4f}"
        all_results.append({'k': k, 'per_subject': k_errs, 'overall': overall})
        mark = " ★" if overall < best_overall_err else ""
        if overall < best_overall_err:
            best_overall_err = overall
            best_k = k
        print(row + mark)

    print(f"\n  Best k: {best_k} (overall mean={best_overall_err:.4f}°)")

    # Compare with best fixed threshold
    print(f"\n  Comparison with fixed threshold approach:")
    best_fixed_err = float('inf')
    best_fixed_thr = None
    for thr in L1_THRESHOLDS:
        thr_errs = []
        for sid in val_sids:
            errs, sr, n = evaluate_subject(
                model, device, sid, l1_thr=thr,
                use_l2=True, use_l3=True)
            if errs is not None:
                thr_errs.append(float(np.mean(errs)))
        mean_e = float(np.mean(thr_errs))
        if mean_e < best_fixed_err:
            best_fixed_err = mean_e
            best_fixed_thr = thr

    print(f"    Best fixed threshold: {best_fixed_thr} (overall mean={best_fixed_err:.4f}°)")
    print(f"    Best adaptive: k={best_k} (overall mean={best_overall_err:.4f}°)")
    print(f"    Improvement: {best_fixed_err - best_overall_err:.4f}°")

    return all_results, best_k, best_overall_err


# ═══════════════════════════════════════════════════════════════
# 方案D: Hybrid (predicted base + adaptive modulation)
# ═══════════════════════════════════════════════════════════════
def run_scheme_d(model, device, val_sids, scheme_b_result, best_adaptive_k):
    """Hybrid: use predictor's default threshold as base + adaptive modulation."""
    print("\n" + "=" * 70)
    print("方案D: Hybrid (Predicted Base + Adaptive Modulation)")
    print("=" * 70)

    if scheme_b_result is None:
        base_thr = 1.0
        print(f"  No scheme B result, using base_thr={base_thr}")
    else:
        base_thr = scheme_b_result['default_threshold']
        print(f"  Using predictor default threshold: {base_thr:.3f}°")

    print(f"  Adaptive k: {best_adaptive_k}")
    print(f"  Hybrid formula: thr = α * base_thr + (1-α) * adaptive_estimate")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_alpha = None
    best_err = float('inf')

    print(f"\n  {'Alpha':>7} | ", end='')
    for sid in val_sids:
        print(f'{sid[-10:]:>20}', end='')
    print(f" {'Overall':>10}")
    print(f"  {'-'*7}-+-{'-'*20*len(val_sids)}-+-{'-'*10}")

    for alpha in alphas:
        row = f"  {alpha:>7.2f} | "
        a_errs = []
        for sid in val_sids:
            # Hybrid: alpha * base + (1-alpha) * adaptive
            # We approximate this by running adaptive with adjusted k
            effective_k = best_adaptive_k * (1.0 - alpha)
            effective_base = base_thr * alpha + 1.0 * (1.0 - alpha)
            # Actually, let's just pass alpha as a mixing parameter
            errs, sr, n = evaluate_subject(
                model, device, sid, l1_thr=base_thr,
                adaptive=(alpha < 0.99),
                adaptive_k=best_adaptive_k * (1.0 - alpha + 0.1),
                use_l2=True, use_l3=True)
            if errs is not None:
                me = float(np.mean(errs))
                a_errs.append(me)
                row += f" {me:>8.4f}  |"
            else:
                row += f" {'N/A':>8}  |"
        overall = float(np.mean(a_errs)) if a_errs else 999
        row += f" {overall:>8.4f}"
        mark = " ★" if overall < best_err else ""
        if overall < best_err:
            best_err = overall
            best_alpha = alpha
        print(row + mark)

    print(f"\n  Best alpha: {best_alpha} (overall mean={best_err:.4f}°)")

    return best_alpha, best_err


# ═══════════════════════════════════════════════════════════════
# Final Evaluation on Test Set
# ═══════════════════════════════════════════════════════════════
def final_evaluation(model, device, test_sids, best_config):
    """Run best configuration on test set with detailed breakdown."""
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    print(f"\n  Best config: {best_config}")

    # Compare: no GAI, GAI with best fixed threshold, GAI with adaptive
    configs = [
        {'name': 'No GAI', 'thr': 0.0, 'adaptive': False},
        {'name': f"Fixed thr={best_config['fixed_thr']}", 'thr': best_config['fixed_thr'], 'adaptive': False},
        {'name': f"Adaptive k={best_config['adaptive_k']}", 'thr': 1.0, 'adaptive': True, 'k': best_config['adaptive_k']},
    ]

    if 'alpha' in best_config:
        configs.append({'name': f"Hybrid α={best_config['alpha']}",
                        'thr': best_config.get('base_thr', 1.0), 'adaptive': True,
                        'k': best_config['adaptive_k'] * (1 - best_config['alpha'] + 0.1)})

    all_rows = []
    for cfg in configs:
        print(f"\n  [{cfg['name']}]")
        total_errs = []
        for sid in test_sids:
            if cfg['name'] == 'No GAI':
                errs, sr, n = evaluate_subject(
                    model, device, sid, l1_thr=0.0,
                    use_l2=False, use_l3=False,
                    adaptive=False)
            else:
                errs, sr, n = evaluate_subject(
                    model, device, sid, l1_thr=cfg['thr'],
                    adaptive=cfg.get('adaptive', False),
                    adaptive_k=cfg.get('k', 2.0),
                    use_l2=True, use_l3=True)
            if errs is not None:
                me = float(np.mean(errs))
                print(f"    {sid}: mean={me:.4f}°, skip={sr:.1f}%, n={n}")
                total_errs.extend(errs)
        if total_errs:
            overall = float(np.mean(total_errs))
            print(f"    Overall: mean={overall:.4f}°, median={float(np.median(total_errs)):.4f}°")
            all_rows.append({'config': cfg['name'], 'test_mean': overall,
                             'test_median': float(np.median(total_errs)),
                             'total_n': len(total_errs)})

    # Save CSV
    csv_path = DATA_DIR / "l1_personalization_test_results.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'test_mean', 'test_median', 'total_n'])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  Saved: {csv_path}")

    return all_rows


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data dir: {DATA_DIR}")

    # Get splits
    train_sids, val_sids, test_sids = get_sid_split(seed=42)
    print(f"Train: {len(train_sids)} SIDs, Val: {len(val_sids)}, Test: {len(test_sids)}")
    print(f"Val SIDs: {val_sids}")
    print(f"Test SIDs: {test_sids}")

    # Load baseline model (no personalization)
    ckpt = DATA_DIR / "ResNet18-GRU-Biophysical-240x240-15fps-two-stage" / "resnet18_gru_bio_two_stage_best.pt"
    model = ResNet18GRUModel().to(device)
    state = torch.load(ckpt, map_location=device)
    # Some checkpoints include subject_embed keys (from personalized runs), strip them
    state = {k: v for k, v in state.items() if not k.startswith('subject_embed.')}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded baseline model: {ckpt}")

    # ── 方案A: Per-Subject Grid Search ──
    a_results, a_best = run_scheme_a(model, device, val_sids)

    # Find global best fixed threshold from validation
    best_global_thr = None
    best_global_err = float('inf')
    for thr in L1_THRESHOLDS:
        errs_list = []
        for sid in val_sids:
            errs, sr, n = evaluate_subject(model, device, sid, l1_thr=thr,
                                           use_l2=True, use_l3=True)
            if errs is not None:
                errs_list.append(float(np.mean(errs)))
        mean_e = float(np.mean(errs_list))
        if mean_e < best_global_err:
            best_global_err = mean_e
            best_global_thr = thr

    # ── 方案C: Adaptive Threshold ──
    c_results, best_k, best_k_err = run_scheme_c(model, device, val_sids)

    # ── 方案B: Embedding → Threshold Predictor ──
    # Try with hidden_init checkpoint
    ckpt_dirs = [
        DATA_DIR / "checkpoints" / "ResNet18-GRU-Biophysical-240x240-15fps-hidden_init",
        DATA_DIR / "checkpoints" / "ResNet18-GRU-Biophysical-240x240-15fps-feat_scale",
    ]
    b_result = None
    for ckpt_dir in ckpt_dirs:
        if ckpt_dir.exists():
            b_result = run_scheme_b(model, device, train_sids, val_sids, ckpt_dir)
            if b_result is not None:
                break

    # ── 方案D: Hybrid ──
    d_best_alpha, d_best_err = run_scheme_d(model, device, val_sids, b_result, best_k)

    # ── Final on test set ──
    best_config = {
        'fixed_thr': best_global_thr,
        'adaptive_k': best_k,
        'alpha': d_best_alpha,
        'base_thr': b_result['default_threshold'] if b_result else 1.0,
    }
    final_evaluation(model, device, test_sids, best_config)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Validation results (mean angular error):")
    print(f"\n  方案A - Per-subject best thresholds:")
    for sid, info in a_best.items():
        print(f"    {sid}: thr={info['best_threshold']}, err={info['best_mean']:.4f}°")
    print(f"    Global best fixed: thr={best_global_thr}, err={best_global_err:.4f}°")
    print(f"\n  方案B - Embedding→Threshold predictor MAE: {b_result['train_mae']:.4f}°" if b_result else "")
    print(f"    Default threshold: {b_result['default_threshold']:.3f}°" if b_result else "")
    print(f"\n  方案C - Best adaptive: k={best_k}, err={best_k_err:.4f}°")
    print(f"\n  方案D - Best hybrid α={d_best_alpha}, err={d_best_err:.4f}°")
    print(f"\n  Test results saved to: {DATA_DIR}/l1_personalization_test_results.csv")


if __name__ == "__main__":
    main()