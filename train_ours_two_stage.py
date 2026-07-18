#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# merge_segments_v2
"""
ResNet-18 + GRU + 生物物理约束 - 二阶段训练 (严格按论文实现，修正版)
=======================================================================
修正内容：
  1. Flip 约束和 C3 速度约束仅计算窗口最后两帧（T-1 与 T），而非整个窗口所有帧。
  2. 添加 ImageNet 归一化，匹配预训练 ResNet 的输入分布。
  3. 其他逻辑与论文完全一致。
"""

import os
import sys
import re
import math
import random
import csv
import datetime
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchvision.transforms import RandomErasing
from tqdm import tqdm

# =====================================================================
# 物理常数
# =====================================================================
EYE_MAX_VEL_DEG      = 700.0   # deg/s 峰值眼跳速度
EYE_CLIP_FPS         = 60.0    # 相机采集帧率（源帧率）
CORNEAL_MAX_ELEV_DEG = 40.0
IRIS_MAX_DEG         = 35.0
LAMBDA_HINGE         = 0.01    # hinge 项独立权重（论文未指定，经验值）
LAMBDA_TEMP          = 0.1     # 时序一致性正则化权重
TEMP_THRESH_DEG      = 1.0     # 时序一致性阈值：仅当 GT 变化 < 阈值时施加约束

# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =====================================================================
# 几何辅助函数
# =====================================================================
def angular_diff_deg(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """向量间角度差（度）"""
    v1n = F.normalize(v1.float(), dim=-1)
    v2n = F.normalize(v2.float(), dim=-1)
    cos = (v1n * v2n).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos) * (180.0 / math.pi)

# =====================================================================
# 生物物理约束损失函数（严格按论文 Sec. 3.1.3，仅用最后两帧）
# =====================================================================
def corneal_pupil_loss(pred: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    C1 角膜-瞳孔约束中与时间无关的部分（式6-8）
    pred: (B, 3) 最后帧预测
    reduction: 'mean' 或 'none'
    """
    pred_n = F.normalize(pred.float(), dim=-1)
    px, py, pz = pred_n[:, 0], pred_n[:, 1], pred_n[:, 2]

    # Elevation constraint (式7)
    cos_max = math.cos(math.radians(CORNEAL_MAX_ELEV_DEG))
    elev = F.relu(cos_max - torch.abs(pz.clamp(-1 + 1e-6, 1 - 1e-6)))  # (B,)

    # Spherical consistency (式8)
    xy_mag = torch.sqrt(px ** 2 + py ** 2 + 1e-8)
    exp_xy = torch.sqrt(torch.clamp(1.0 - pz ** 2, min=0.0) + 1e-8)
    xy_inc = F.relu(xy_mag - exp_xy - 0.1)  # (B,)

    per_sample = elev + 0.5 * xy_inc  # (B,)
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


def corneal_flip_loss_pair(pred_seq: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    C1 中的 Flip 约束（式9），仅计算最后两帧（T-1 与 T）
    pred_seq: (B, T, 3)
    reduction: 'mean' 或 'none'
    """
    if pred_seq.shape[1] < 2:
        if reduction == 'mean':
            return torch.zeros(1, device=pred_seq.device, requires_grad=True)[0]
        return torch.zeros(pred_seq.shape[0], device=pred_seq.device, requires_grad=True)
    # 取最后两帧的 z 分量
    pz_tm1 = F.normalize(pred_seq[:, -2], dim=-1)[..., 2]   # (B,)
    pz_t   = F.normalize(pred_seq[:, -1], dim=-1)[..., 2]
    per_sample = 0.2 * F.relu(-pz_tm1 * pz_t)  # (B,)
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


def iris_boundary_loss(pred: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    C2 虹膜边界约束（式10-11），作用于最后帧
    pred: (B, 3)
    reduction: 'mean' 或 'none'
    """
    pred_n = F.normalize(pred.float(), dim=-1)
    fwd = torch.zeros(3, device=pred.device); fwd[2] = 1.0
    cos = (pred_n * fwd).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos) * (180.0 / math.pi)
    per_sample = F.relu(angle - IRIS_MAX_DEG)  # (B,)
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


def neuro_mechanical_loss_pair(pred_seq: torch.Tensor,
                               target_seq: torch.Tensor,
                               effective_fps: float,
                               reduction: str = 'mean') -> torch.Tensor:
    """
    C3 速度连续性约束（式12,14,15,17），仅计算最后两帧
    pred_seq, target_seq: (B, T, 3)
    effective_fps: 有效帧率
    reduction: 'mean' 或 'none'
    """
    B = pred_seq.shape[0]
    device = pred_seq.device
    if pred_seq.shape[1] < 2:
        if reduction == 'mean':
            return torch.zeros(1, device=device, requires_grad=True)[0]
        return torch.zeros(B, device=device, requires_grad=True)

    theta_max = EYE_MAX_VEL_DEG / effective_fps   # 每帧允许最大角度 (deg)

    # 预测最后两帧的角度差
    p_tm1 = F.normalize(pred_seq[:, -2], dim=-1)
    p_t   = F.normalize(pred_seq[:, -1], dim=-1)
    cos_p = (p_tm1 * p_t).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    ang_p = torch.acos(cos_p) * (180.0 / math.pi)   # (B,)

    # 真值最后两帧的角度差
    t_tm1 = F.normalize(target_seq[:, -2], dim=-1)
    t_t   = F.normalize(target_seq[:, -1], dim=-1)
    cos_t = (t_tm1 * t_t).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    ang_t = torch.acos(cos_t) * (180.0 / math.pi)   # (B,)

    # L_vel (式15) + L_smooth (式17)
    per_sample = F.relu(ang_p - theta_max) + 0.3 * F.relu(ang_p - ang_t - theta_max)  # (B,)
    # NaN guard: replace NaN samples with 0
    per_sample = torch.where(torch.isnan(per_sample), torch.zeros_like(per_sample), per_sample)
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


def c3_hinge_loss(pred: torch.Tensor,
                  target: torch.Tensor,
                  effective_fps: float,
                  reduction: str = 'mean') -> torch.Tensor:
    """
    Hinge 辅助项 (论文 Eq.18)：对最后帧 pred-GT 角度超出 5×θ_max_vel 的惩罚。
    reduction: 'mean' 或 'none'
    """
    theta_max = EYE_MAX_VEL_DEG / effective_fps
    ang_dev = angular_diff_deg(pred, target)   # (B,)
    per_sample = F.relu(ang_dev - 5.0 * theta_max)  # (B,)
    per_sample = torch.where(torch.isnan(per_sample), torch.zeros_like(per_sample), per_sample)
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


# =====================================================================
# 模型定义（支持返回整个序列的输出）
# =====================================================================
class ResNet18GRUModel(nn.Module):
    """ResNet-18 + 2层GRU，可输出全序列或最后一帧

    Args:
        personalize_mode: 'none' | 'feat_scale' | 'hidden_init'
        num_subjects: 训练受试者总数（personalize 时需要）
    """
    def __init__(self, feat_dim: int = 512, hidden: int = 256, dropout: float = 0.3,
                 personalize_mode: str = 'none', num_subjects: int = 0,
                 pretrained_backbone: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_drop = nn.Dropout(dropout)
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden,
                          num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )
        self.personalize_mode = personalize_mode
        if personalize_mode != 'none' and num_subjects > 0:
            self.subject_embed = SubjectEmbedding(num_subjects, feat_dim=feat_dim, hidden=hidden)

    def forward(self, x, subject_idx=None):
        """Return the final prediction, including personalization when supplied."""
        return self.forward_all(x, subject_idx=subject_idx)[:, -1]

    def forward_all(self, x, use_cache_aug=False, subject_idx=None):
        """返回整个序列的预测 (B, T, 3)

        Args:
            subject_idx: (B,) 训练受试者索引，为 None 时跳过个性化
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W).contiguous()
        f = self.backbone(x)
        f = self.feat_drop(f.flatten(1))
        f = f.reshape(B, T, -1)

        # ── 个性化：特征缩放 ──
        if self.personalize_mode == 'feat_scale' and subject_idx is not None:
            valid = (subject_idx >= 0)
            if valid.all():
                f = f * self.subject_embed.forward_feat_scale(subject_idx)
            elif valid.any():
                scale = torch.ones(B, 1, self.feat_dim, device=f.device)
                scale[valid] = self.subject_embed.forward_feat_scale(subject_idx[valid])
                f = f * scale

        # 特征缓存增强：模拟推理时跳 ResNet 的场景
        if use_cache_aug and self.training and random.random() < 0.3:
            cache_start = random.randint(1, T - 2)
            cache_len = min(random.randint(1, 5), T - cache_start)
            with torch.no_grad():
                f[:, cache_start:cache_start+cache_len] = f[:, cache_start-1:cache_start]

        # ── 个性化：GRU 隐状态初始化 ──
        if self.personalize_mode == 'hidden_init' and subject_idx is not None:
            valid = (subject_idx >= 0)
            if valid.all():
                h0 = self.subject_embed.forward_hidden_init(subject_idx, B, f.device)
            elif valid.any():
                h0 = torch.zeros(2, B, self.hidden, device=f.device)
                h0[:, valid] = self.subject_embed.forward_hidden_init(
                    subject_idx[valid], valid.sum(), f.device)
            else:
                h0 = None
            out, _ = self.gru(f, h0)
        else:
            out, _ = self.gru(f)
        return self.head(out)

    def forward_features(self, x):
        """仅提取 ResNet 特征，不跑 GRU。返回 (B, 512) 最后一帧特征。"""
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W).contiguous()
        f = self.backbone(x)
        f = self.feat_drop(f.flatten(1))
        f = f.reshape(B, T, -1)
        return f[:, -1]  # (B, 512)

    def forward_with_hidden(self, x):
        """全量推理，返回 (pred, hidden_state, last_features)。
        pred: (B, 3), hidden: (2, B, 256), features: (B, 512)
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W).contiguous()
        f = self.backbone(x)
        f = self.feat_drop(f.flatten(1))
        f = f.reshape(B, T, -1)
        out, hidden = self.gru(f)
        return self.head(out[:, -1]), hidden, f[:, -1]

    def forward_cached(self, features, hidden_state):
        """跳过 ResNet，用缓存特征跑 GRU+head。
        features: (B, 1, 512), hidden_state: (2, B, 256)
        返回 (pred, new_hidden)
        """
        out, new_hidden = self.gru(features, hidden_state)
        return self.head(out[:, -1]), new_hidden


# =====================================================================
# 数据集：返回整个窗口的 GT 序列，并添加 ImageNet 归一化
# =====================================================================
def load_gaze_vec(path: Path) -> np.ndarray:
    arr = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 4:
                continue
            try:
                arr.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except:
                continue
    return np.asarray(arr, dtype=np.float32)


def load_validity(path: Path) -> np.ndarray:
    arr = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
            try:
                arr.append(int(parts[1]))
            except:
                continue
    return np.asarray(arr, dtype=np.int32)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def detect_concat_points(dir_path, sid, z_thresh=10.0, min_gap=300, fps=60.0):
    lmpath = dir_path / f"{sid}.mp4pupil_lm_3D.txt"
    valpath = dir_path / f"{sid}.mp4validity_pupil.txt"
    lm = []
    with lmpath.open("r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 5:
                lm.append((np.nan, np.nan, np.nan))
                continue
            nums = []
            for p in parts[2:]:
                if p == "":
                    continue
                try:
                    nums.append(float(p))
                except:
                    nums.append(np.nan)
            nums = np.asarray(nums, dtype=np.float32)
            k = nums.size // 3
            if k == 0:
                lm.append((np.nan, np.nan, np.nan))
                continue
            xyz = nums[:k * 3].reshape(k, 3)
            lm.append(tuple(np.nanmean(xyz, axis=0).tolist()))
    lm = np.asarray(lm, dtype=np.float32)
    valid = load_validity(valpath)
    n = min(len(lm), len(valid))
    lm = lm[:n]
    valid = valid[:n]
    diff = np.full(n, np.nan, dtype=np.float32)
    diff[1:] = np.linalg.norm(lm[1:] - lm[:-1], axis=1)
    diff[valid == 0] = np.nan
    z = robust_zscore(diff)
    cand = np.where(z > z_thresh)[0].tolist()
    cuts, last = [], -10**9
    for c in cand:
        if c - last >= min_gap:
            cuts.append(int(c))
            last = c
    segs, prev = [], 0
    for c in cuts:
        if c > prev:
            segs.append((prev, c, c - prev, (c - prev) / fps))
        prev = c
    if prev < n:
        segs.append((prev, n, n - prev, (n - prev) / fps))
    return cuts, segs


def merge_segments_v2(segs, min_len=600, drop_tiny=120):
    """
    保留由 detect_concat_points() 得到的真实拼接边界，避免把断点两侧的 segment 重新合并。

    旧版本会把短 segment 与后一个 segment 合并，这可能跨越 concat cut，导致窗口包含不连续帧，
    从而破坏 C3 速度连续性约束。这里改为：
      1. 丢弃极短 segment；
      2. 不跨边界合并 segment；
      3. 只保留长度达到 min_len 的 segment。
    """
    kept = []
    for a, b, L, _ in segs:
        if L < drop_tiny:
            continue
        if L < min_len:
            continue
        kept.append((a, b, L, L / 60.0))
    return kept


class TEyeDSeqDataset(Dataset):
    """
    返回 (clip_x, gaze_seq)
    clip_x: (T, 3, H, W)  视频片段，已应用 ImageNet 归一化
    gaze_seq: (T, 3) 对应每一帧的 ground-truth gaze 向量
    """
    def __init__(self, data_dir: str, sids: List[str],
                 clip_len: int = 8, stride: int = 4, img_size: int = 128,
                 z_thresh: float = 10.0, min_gap: int = 300,
                 segment_min_len: int = 600,
                 max_segments_per_video: int = 60,
                 max_clips_per_segment: int = 80,
                 require_all_valid: bool = True,
                 augment: bool = False,
                 seed: int = 42,
                 sid_to_idx=None):
        self.data_dir = Path(data_dir)
        self.clip_len = clip_len
        self.stride = stride
        self.img_size = img_size
        self.fps = EYE_CLIP_FPS
        self.rng = random.Random(seed)
        self.require_all_valid = require_all_valid
        self.augment = augment
        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.sid_to_idx = sid_to_idx or {}

        self.items: List[Tuple[str, int]] = []  # (sid, start_frame)
        self.meta: Dict[str, dict] = {}

        for sid in sids:
            vpath = self.data_dir / f"{sid}.mp4pupil_seg_3D.mp4"
            gpath = self.data_dir / f"{sid}.mp4gaze_vec.txt"
            valpath = self.data_dir / f"{sid}.mp4validity_pupil.txt"
            lmpath = self.data_dir / f"{sid}.mp4pupil_lm_3D.txt"
            if not all(p.exists() for p in [vpath, gpath, valpath, lmpath]):
                continue
            gaze = load_gaze_vec(gpath)
            valid = load_validity(valpath).astype(np.int32)
            n = min(len(gaze), len(valid))
            gaze = gaze[:n]
            valid = valid[:n]
            bad = np.all(gaze == -1.0, axis=1)
            valid[bad] = 0

            _, segs = detect_concat_points(self.data_dir, sid, z_thresh, min_gap, self.fps)
            merged = merge_segments_v2(segs, segment_min_len)
            if max_segments_per_video and len(merged) > max_segments_per_video:
                merged = self.rng.sample(merged, k=max_segments_per_video)
                merged.sort(key=lambda x: x[0])

            need_frames = (clip_len - 1) * stride + 1
            for (a, b, L, _) in merged:
                if b - a < need_frames:
                    continue
                starts = list(range(a, b - need_frames + 1, stride))
                ok_starts = []
                for s in starts:
                    idxs = np.arange(s, s + need_frames, stride, dtype=np.int64)
                    if idxs.max() >= len(valid):
                        continue
                    if (not self.require_all_valid or
                            np.all(valid[s:s + need_frames] == 1)):
                        ok_starts.append(s)
                if not ok_starts:
                    continue
                k = min(max_clips_per_segment, len(ok_starts))
                positions = np.linspace(0, len(ok_starts) - 1, k, dtype=np.int64)
                for position in positions:
                    s = ok_starts[int(position)]
                    self.items.append((sid, int(s)))

            self.meta[sid] = {
                "video_path": str(vpath),
                "gaze": gaze,
                "valid": valid
            }

        self.items.sort(key=lambda x: (x[0], x[1]))
        print(f"[TEyeDSeqDataset] SIDs={len(self.meta)}, clips={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, start = self.items[idx]
        vpath = self.meta[sid]["video_path"]
        gaze = self.meta[sid]["gaze"]

        need_frames = (self.clip_len - 1) * self.stride + 1
        start_sec = start / self.fps
        end_sec = (start + need_frames) / self.fps
        frames, _, _ = read_video(vpath,
                                  start_pts=start_sec,
                                  end_pts=end_sec,
                                  pts_unit="sec")
        if frames.shape[0] < need_frames:
            frames = torch.cat([frames, frames[-1:].repeat(need_frames - frames.shape[0], 1, 1, 1)])
        elif frames.shape[0] > need_frames:
            frames = frames[:need_frames]

        # 采样 stride 帧
        frames = frames[::self.stride]                     # (T, H, W, 3)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0   # (T, 3, H, W)
        if self.img_size:
            frames = F.interpolate(frames, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)
        # 应用 ImageNet 归一化
        frames = self.norm(frames)

        # 获取对应每帧的 gaze 向量
        frame_indices = np.arange(start, start + need_frames, self.stride, dtype=np.int64)
        gaze_seq = gaze[frame_indices]   # (T, 3)
        gaze_seq = torch.from_numpy(gaze_seq).float()

        # ---------- 数据增强（仅训练时）----------
        if self.augment:
            # a) Random Horizontal Flip (p=0.5)：所有 T 帧同时翻转
            if random.random() < 0.5:
                frames = torch.flip(frames, [-1])          # (T, 3, H, W) 水平翻转
                gaze_seq[:, 0] *= -1.0                      # gaze x 分量取反

            # b) Random Erasing (p=0.25)：对每帧独立应用
            if random.random() < 0.25:
                eraser = RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3))
                frames = eraser(frames)

        # Return subject index for personalized lambdas (or -1 if unknown)
        subject_idx = self.sid_to_idx.get(sid, -1)

        return frames, gaze_seq, subject_idx


# =====================================================================
# EMA（Exponential Moving Average）模型参数平均
# =====================================================================
class EMA:
    """模型参数的指数滑动平均，推理时用滑动平均版本通常比原始权重更稳定。"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """每个 optimizer.step() 后调用，更新滑动平均。"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        """验证/测试前调用，切换到 EMA 权重。"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        """验证/测试后调用，恢复原始权重继续训练。"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

    def state_dict(self, model):
        """Build a complete model state dict with EMA parameters and live buffers."""
        state = {name: value.detach().clone()
                 for name, value in model.state_dict().items()}
        for name, value in self.shadow.items():
            state[name] = value.detach().clone()
        return state


# =====================================================================
# Subject Embedding 个性化
# =====================================================================
class SubjectEmbedding(nn.Module):
    """受试者 embedding，通过调制特征或 GRU 隐状态实现个性化。

    Mode 'feat_scale': embedding → Linear(16, 512) → 缩放 ResNet 特征
    Mode 'hidden_init': embedding → Linear(16, 512) → GRU 隐状态初始化
    """
    def __init__(self, num_subjects: int, embed_dim: int = 16, feat_dim: int = 512,
                 hidden: int = 256, num_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_subjects, embed_dim)
        # 特征缩放投影 (for feat_scale)
        self.feat_proj = nn.Linear(embed_dim, feat_dim)
        # GRU 隐状态初始化投影 (for hidden_init)
        self.hidden_proj = nn.Linear(embed_dim, hidden * num_layers)
        # A new subject must initially reproduce the universal model exactly.
        nn.init.zeros_(self.feat_proj.weight)
        nn.init.zeros_(self.feat_proj.bias)
        nn.init.zeros_(self.hidden_proj.weight)
        nn.init.zeros_(self.hidden_proj.bias)

    def forward_feat_scale(self, subject_idx):
        """返回特征缩放因子 (B, 1, 512)"""
        emb = self.embedding(subject_idx)  # (B, 16)
        scale = self.feat_proj(emb)        # (B, 512)
        return 1.0 + scale.unsqueeze(1)    # (B, 1, 512)

    def forward_hidden_init(self, subject_idx, batch_size, device):
        """返回 GRU 初始隐状态 (num_layers, B, hidden)"""
        emb = self.embedding(subject_idx)  # (B, 16)
        h0 = self.hidden_proj(emb)         # (B, 512)
        # Projected values are laid out per subject, then per GRU layer.
        return (h0.reshape(batch_size, self.num_layers, self.hidden)
                .permute(1, 0, 2).contiguous())

    def zero_scale(self, batch_size, dim, device):
        """未见受试者使用零调制（scale=1）"""
        return torch.ones(batch_size, 1, dim, device=device)

    def zero_hidden(self, num_layers, batch_size, hidden, device):
        """未见受试者使用零隐状态"""
        return torch.zeros(num_layers, batch_size, hidden, device=device)


# =====================================================================
# 二阶段训练函数（严格按论文）
# =====================================================================
def train_two_stage(
    data_dir: str = os.environ.get(
        "EYE_DATA_DIR", "/home/luxliang/datasets/EXPORT_PUPIL_ALL"),
    clip_len: int = 8,
    stride: int = 4,
    img_size: int = 240,
    batch_size: int = 16,
    epochs_stage1: int = 6,
    epochs_stage2: int = 9,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 10,
    augment: bool = True,
    seed: int = 42,
    personalize_mode: str = 'none',
    output_dir: Optional[str] = None,
    subject_split: Optional[Dict[str, List[str]]] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    effective_fps = EYE_CLIP_FPS / stride
    fps_tag = int(round(effective_fps))
    output_dir = (Path(output_dir) if output_dir else
                  Path("checkpoints") /
                  f"ResNet18-GRU-Biophysical-{img_size}x{img_size}-{fps_tag}fps-{personalize_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lambda_c1_stage1 = 0.1
    lambda_c2_stage1 = 0.3
    lambda_c3_stage1 = 0.2
    lambda_c1_stage2 = 0.001
    lambda_c2_stage2 = 0.001
    lambda_c3_stage2 = 0.01

    print("=" * 80)
    print("ResNet-18 + GRU + Biophysical - 二阶段训练 (严格按论文，约束仅用最后两帧)")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"有效帧率: {effective_fps:.1f} fps  (stride={stride})")
    print(f"θ_max_vel = {EYE_MAX_VEL_DEG / effective_fps:.2f}° per frame")
    print(f"阶段1: {epochs_stage1} epochs, λ_C1=0.1, λ_C2=0.3, λ_C3=0.2")
    print(f"阶段2: {epochs_stage2} epochs, λ_C1=0.001, λ_C2=0.001, λ_C3=0.01")
    print(f"Hinge 独立权重: {LAMBDA_HINGE} (论文未指定，经验值)")
    print(f"个性化模式: {personalize_mode}\n")

    # 划分 SID
    root = Path(data_dir)
    pat = re.compile(r"^(NVIDIA(?:AR|VR)_\d+_1)\.mp4pupil_seg_3D\.mp4$")
    all_sids = sorted({pat.match(p.name).group(1)
                       for p in root.iterdir()
                       if p.is_file() and pat.match(p.name)})
    if subject_split is None:
        rng = random.Random(seed)
        shuffled = list(all_sids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(round(n * 0.1)))
        test_sids = shuffled[n - n_test:]
        train_val_sids = shuffled[:n - n_test]
        rng.shuffle(train_val_sids)
        n_val = max(1, int(round(len(train_val_sids) * 0.1)))
        val_sids = train_val_sids[:n_val]
        train_sids = train_val_sids[n_val:]
    else:
        train_sids = list(subject_split["train_sids"])
        val_sids = list(subject_split["val_sids"])
        test_sids = list(subject_split["test_sids"])
        split_sets = [set(train_sids), set(val_sids), set(test_sids)]
        if any(not split for split in split_sets):
            raise ValueError("Explicit train/val/test subject splits must be non-empty")
        if any(split_sets[i].intersection(split_sets[j])
               for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Explicit train/val/test subject splits overlap")
        unknown = sorted(set().union(*split_sets) - set(all_sids))
        if unknown:
            raise ValueError(f"Unknown SIDs in explicit split: {unknown}")
    print(f"训练 SIDs: {len(train_sids)}, 验证 SIDs: {len(val_sids)}, 测试 SIDs: {len(test_sids)}")
    with (output_dir / "subject_split.json").open("w") as handle:
        json.dump({
            "seed": seed,
            "train_sids": train_sids,
            "val_sids": val_sids,
            "test_sids": test_sids,
        }, handle, indent=2)

    # 受试者索引映射（始终构建，用于 dataset 返回 subject_idx）
    sid_to_idx = {sid: i for i, sid in enumerate(train_sids)} if personalize_mode != 'none' else {}

    # 数据集（返回全序列 GT）
    ds_kw = dict(
        clip_len=clip_len, stride=stride, img_size=img_size,
        z_thresh=10.0, min_gap=300,
        segment_min_len=int(os.environ.get("EYE_SEG_MIN_LEN", "120")),
        max_segments_per_video=60,
        max_clips_per_segment=int(os.environ.get("EYE_MAX_CLIPS_PER_SEG", "15")),
        require_all_valid=True, seed=seed
    )
    ds_train = TEyeDSeqDataset(data_dir, train_sids, augment=augment, sid_to_idx=sid_to_idx, **ds_kw)
    ds_val   = TEyeDSeqDataset(data_dir, val_sids,   **ds_kw)
    ds_test  = TEyeDSeqDataset(data_dir, test_sids,  **ds_kw)

    tr_loader = DataLoader(ds_train, batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size, shuffle=False, num_workers=4)
    te_loader = DataLoader(ds_test, batch_size, shuffle=False, num_workers=4)

    model = ResNet18GRUModel(personalize_mode=personalize_mode,
                             num_subjects=len(train_sids)).to(device)
    ema = EMA(model, decay=0.999)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {params:.2f} M")
    print(f"EMA: 已启用 (decay=0.999)")
    aug_status = "开启" if augment else "关闭"
    print(f"数据增强: {aug_status}")
    print(f"LR Warmup: 第一个 epoch 线性预热\n")

    # =================================================================
    # 阶段1
    # =================================================================
    print("\n" + "=" * 80)
    print("【阶段1】预训练 - 大权重生物约束")
    print("=" * 80)

    optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=3)

    best_val = float('inf')
    best_path_stage1 = output_dir / "stage1_best.pt"
    no_improve = 0

    # Stage 1: 总批次数（用于 LR warmup）
    s1_total_batches = len(tr_loader)

    for epoch in range(1, epochs_stage1 + 1):
        model.train()
        train_losses = []
        for batch_idx, (x, y_seq, subject_idx) in enumerate(tqdm(tr_loader, desc=f"Stage1 Ep{epoch}/{epochs_stage1}")):
            # LR Warmup: 第一个 epoch 线性从 0 升到 lr
            if epoch == 1:
                warmup_step = batch_idx / s1_total_batches
                current_lr = lr * min(1.0, warmup_step)
                for pg in optimizer1.param_groups:
                    pg['lr'] = current_lr

            x = x.to(device)
            y_seq = y_seq.to(device)
            subject_idx = subject_idx.to(device)

            pred_seq = model.forward_all(x, use_cache_aug=True, subject_idx=subject_idx)  # (B, T, 3)
            pred_last = pred_seq[:, -1]       # (B, 3)
            y_last = y_seq[:, -1]             # (B, 3)

            # 主损失
            loss = F.mse_loss(pred_last, y_last)
            loss += ((pred_last.norm(dim=-1) - 1.0) ** 2).mean()

            # C1: 时不变部分 + 仅最后两帧的 Flip
            loss += lambda_c1_stage1 * corneal_pupil_loss(pred_last)
            loss += lambda_c1_stage1 * corneal_flip_loss_pair(pred_seq)

            # C2
            loss += lambda_c2_stage1 * iris_boundary_loss(pred_last)

            # C3: 仅最后两帧的速度约束
            loss += lambda_c3_stage1 * neuro_mechanical_loss_pair(pred_seq, y_seq, effective_fps)

            # Hinge
            loss += LAMBDA_HINGE * c3_hinge_loss(pred_last, y_last, effective_fps)

            # A: 时序一致性正则化 — 注视期相邻帧预测变化应接近零
            temp_loss = 0.0
            for t in range(1, pred_seq.shape[1]):
                gt_change = angular_diff_deg(y_seq[:, t], y_seq[:, t-1])
                pred_change = F.mse_loss(pred_seq[:, t], pred_seq[:, t-1], reduction='none').mean(dim=-1)
                mask = (gt_change < TEMP_THRESH_DEG).float()
                temp_loss += (mask * pred_change).mean()
            temp_loss = temp_loss / (pred_seq.shape[1] - 1)
            loss += LAMBDA_TEMP * temp_loss

            optimizer1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer1.step()
            ema.update(model)
            train_losses.append(loss.item())

        # 验证（使用 EMA 权重）
        ema.apply_shadow(model)
        model.eval()
        val_errs = []
        with torch.no_grad():
            for x, y_seq, _ in val_loader:
                x = x.to(device)
                pred_last = model(x)
                y_last = y_seq[:, -1].to(device)
                errs = angular_diff_deg(pred_last, y_last).cpu().numpy()
                val_errs.extend(errs)
        ema.restore(model)
        val_mean = float(np.mean(val_errs)) if val_errs else float('inf')
        scheduler1.step(val_mean)

        if val_mean < best_val:
            best_val = val_mean
            no_improve = 0
            torch.save(ema.state_dict(model), best_path_stage1)
            mark = " ★"
        else:
            no_improve += 1
            mark = ""
        print(f"[Ep{epoch}] train_loss={np.mean(train_losses):.4f} | val_ang={val_mean:.3f}° | best={best_val:.3f}°{mark}")

        if no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\n✅ 阶段1完成！最佳验证误差: {best_val:.3f}°")

    # =================================================================
    # 阶段2
    # =================================================================
    print("\n" + "=" * 80)
    print("【阶段2】微调 - 小权重生物约束")
    print("=" * 80)
    model.load_state_dict(torch.load(best_path_stage1, map_location=device))
    # 重新初始化 EMA（模型权重已变）
    ema = EMA(model, decay=0.999)
    print(f"Loaded stage-1 best model: {best_path_stage1}\n")

    # 优化器（embedding 参数已包含在 model.parameters() 中）
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=weight_decay)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_stage2, eta_min=1e-6)

    best_val2 = float('inf')
    best_path_stage2 = output_dir / "resnet18_gru_bio_two_stage_best.pt"
    no_improve2 = 0

    s2_total_batches = len(tr_loader)

    for epoch in range(1, epochs_stage2 + 1):
        model.train()
        train_losses = []
        for batch_idx, (x, y_seq, subject_idx) in enumerate(tqdm(tr_loader, desc=f"Stage2 Ep{epoch}/{epochs_stage2}")):
            # LR Warmup: 第一个 epoch 线性从 0 升到 lr*0.5
            if epoch == 1:
                warmup_step = batch_idx / s2_total_batches
                current_lr = (lr * 0.5) * min(1.0, warmup_step)
                for pg in optimizer2.param_groups:
                    pg['lr'] = current_lr

            x = x.to(device)
            y_seq = y_seq.to(device)
            subject_idx = subject_idx.to(device)

            pred_seq = model.forward_all(x, use_cache_aug=True, subject_idx=subject_idx)
            pred_last = pred_seq[:, -1]
            y_last = y_seq[:, -1]

            loss = F.mse_loss(pred_last, y_last)
            loss += ((pred_last.norm(dim=-1) - 1.0) ** 2).mean()

            # ── 约束损失（使用全局 λ） ──
            loss += lambda_c1_stage2 * corneal_pupil_loss(pred_last)
            loss += lambda_c1_stage2 * corneal_flip_loss_pair(pred_seq)
            loss += lambda_c2_stage2 * iris_boundary_loss(pred_last)
            loss += lambda_c3_stage2 * neuro_mechanical_loss_pair(pred_seq, y_seq, effective_fps)

            loss += LAMBDA_HINGE * c3_hinge_loss(pred_last, y_last, effective_fps)

            # A: 时序一致性正则化
            temp_loss = 0.0
            for t in range(1, pred_seq.shape[1]):
                gt_change = angular_diff_deg(y_seq[:, t], y_seq[:, t-1])
                pred_change = F.mse_loss(pred_seq[:, t], pred_seq[:, t-1], reduction='none').mean(dim=-1)
                mask = (gt_change < TEMP_THRESH_DEG).float()
                temp_loss += (mask * pred_change).mean()
            temp_loss = temp_loss / (pred_seq.shape[1] - 1)
            loss += LAMBDA_TEMP * temp_loss

            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer2.step()
            ema.update(model)
            train_losses.append(loss.item())

        scheduler2.step()

        # 验证（使用 EMA 权重）
        ema.apply_shadow(model)
        model.eval()
        val_errs = []
        with torch.no_grad():
            for x, y_seq, _ in val_loader:
                x = x.to(device)
                pred_last = model(x)
                y_last = y_seq[:, -1].to(device)
                val_errs.extend(angular_diff_deg(pred_last, y_last).cpu().numpy())
        ema.restore(model)
        val_mean = float(np.mean(val_errs)) if val_errs else float('inf')

        if val_mean < best_val2:
            best_val2 = val_mean
            no_improve2 = 0
            torch.save(ema.state_dict(model), best_path_stage2)
            mark = " ★"
        else:
            no_improve2 += 1
            mark = ""
        print(f"[Ep{epoch}] train_loss={np.mean(train_losses):.4f} | val_ang={val_mean:.3f}° | best={best_val2:.3f}°{mark}")

        if no_improve2 >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\n✅ 阶段2完成！最佳验证误差: {best_val2:.3f}°")

    # 最终测试（含 TTA：水平翻转集成）
    print("\n" + "=" * 80)
    print("【最终测试 - 含 TTA】")
    print("=" * 80)
    model.load_state_dict(torch.load(best_path_stage2, map_location=device))
    model.eval()
    test_errs = []
    test_errs_tta = []
    with torch.no_grad():
        for x, y_seq, _ in te_loader:
            x = x.to(device)
            y_last = y_seq[:, -1].to(device)

            # 原始推理
            pred_orig = model(x)
            errs = angular_diff_deg(pred_orig, y_last).cpu().numpy()
            test_errs.extend(errs)

            # TTA: 水平翻转集成
            x_flip = torch.flip(x, [-1])            # (B, T, 3, H, W) 水平翻转
            pred_flip = model(x_flip)
            pred_flip[:, 0] *= -1.0                  # gaze x 分量取反还原
            pred_tta = (pred_orig + pred_flip) / 2.0
            errs_tta = angular_diff_deg(pred_tta, y_last).cpu().numpy()
            test_errs_tta.extend(errs_tta)

    test_mean = float(np.mean(test_errs))
    test_med = float(np.median(test_errs))
    test_p90 = float(np.percentile(test_errs, 90))
    tta_mean = float(np.mean(test_errs_tta))
    tta_med = float(np.median(test_errs_tta))
    tta_p90 = float(np.percentile(test_errs_tta, 90))
    print(f"无 TTA: mean={test_mean:.3f}° | median={test_med:.3f}° | p90={test_p90:.3f}°")
    print(f"有 TTA: mean={tta_mean:.3f}° | median={tta_med:.3f}° | p90={tta_p90:.3f}°")
    print(f"TTA 改善: {test_mean - tta_mean:.3f}° ({((test_mean - tta_mean)/test_mean)*100:.1f}%)")

    # 保存结果
    csv_path = output_dir / "two_stage_results.csv"
    param_count = sum(p.numel() for p in model.parameters())
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'experiment', 'model_variant',
                                               'personalize_mode',
                                               'clip_len', 'stride', 'img_size',
                                               'stage1_best', 'stage2_best',
                                               'test_mean', 'test_median', 'test_p90',
                                               'tta_mean', 'tta_median', 'tta_p90',
                                               'parameters'])
        if write_header:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment': f'{img_size}x{img_size}_{fps_tag}fps',
            'model_variant': 'resnet18_gru_bio_two_stage',
            'personalize_mode': personalize_mode,
            'clip_len': clip_len,
            'stride': stride,
            'img_size': img_size,
            'stage1_best': f"{best_val:.4f}",
            'stage2_best': f"{best_val2:.4f}",
            'test_mean': f"{test_mean:.4f}",
            'test_median': f"{test_med:.4f}",
            'test_p90': f"{test_p90:.4f}",
            'tta_mean': f"{tta_mean:.4f}",
            'tta_median': f"{tta_med:.4f}",
            'tta_p90': f"{tta_p90:.4f}",
            'parameters': f"{param_count}"
        })

    print(f"\n✅ 结果已保存: {csv_path}")
    print(f"✅ 最佳模型: {best_path_stage2}")

    # 导出学习的 subject embedding（如果启用个性化）
    if personalize_mode != 'none' and hasattr(model, 'subject_embed'):
        embed_weights = model.subject_embed.embedding.weight.detach().cpu().numpy()  # (N, 16)
        embed_csv = output_dir / "subject_embeddings.csv"
        with open(embed_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sid'] + [f'emb_{i}' for i in range(embed_weights.shape[1])])
            for i, sid in enumerate(train_sids):
                w.writerow([sid] + [f"{v:.6f}" for v in embed_weights[i]])
        print(f"Subject embeddings 已导出: {embed_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ResNet-18 + GRU + Biophysical constraints - two-stage training")
    parser.add_argument("--data_dir", type=str, default=os.environ.get(
        "EYE_DATA_DIR", "/home/luxliang/datasets/EXPORT_PUPIL_ALL"))
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_stage1", type=int, default=6)
    parser.add_argument("--epochs_stage2", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--no-augment", action="store_true", help="禁用数据增强")
    parser.add_argument("--personalize-mode", type=str, default='none',
                        choices=['none', 'feat_scale', 'hidden_init'],
                        help="个性化模式: none | feat_scale | hidden_init")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录；默认按个性化模式隔离到 checkpoints/")
    parser.add_argument("--split-json", type=str, default=None,
                        help="显式用户划分 JSON；可为单个 split 或包含 folds 的文件")
    parser.add_argument("--fold-index", type=int, default=None,
                        help="当 --split-json 包含 folds 时选择的 fold 下标")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subject_split = None
    if args.split_json:
        with open(args.split_json, "r") as handle:
            split_payload = json.load(handle)
        if "folds" in split_payload:
            if args.fold_index is None:
                parser.error("--fold-index is required when --split-json contains folds")
            try:
                subject_split = split_payload["folds"][args.fold_index]
            except IndexError:
                parser.error(f"Invalid --fold-index {args.fold_index}")
        else:
            subject_split = split_payload

    train_two_stage(
        data_dir=args.data_dir,
        augment=not args.no_augment,
        personalize_mode=args.personalize_mode,
        clip_len=args.clip_len,
        stride=args.stride,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        output_dir=args.output_dir,
        subject_split=subject_split,
    )
