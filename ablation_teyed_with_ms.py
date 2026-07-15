#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEyeD消融实验 + 个性化Main Sequence约束集成版
对比：有/无 个性化MS约束 的效果差异
用法：
  python3 ablation_teyed_with_ms.py full     # 完整方法 + MS约束
  python3 ablation_teyed_with_ms.py full_no_ms  # 完整方法，无MS约束（对照组）
"""
import os, sys, math, random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 视频解码：优先用decord（帧精确、高效），备选torchvision read_video
try:
    from decord import VideoReader as DecordVR, cpu as decord_cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    from torchvision.io import read_video

# 导入个性化MS约束模块
from personalized_main_sequence import (
    PersonalizedGazeLoss,
    UserMainSequenceBank,
    SaccadeDetector,
    MainSequenceCalibrator,
    POPULATION_PRIOR,
)

# ============================================================
# 超参配置（可用环境变量覆盖，便于烟囱测试/迁移不同机器）
#   GAZE_DATA_DIR  数据目录（默认 TEyeD 导出目录）
#   GAZE_EPOCHS    总训练轮数
#   GAZE_MAX_VIDEOS  仅用前 N 个视频（烟囱测试用；0=全部）
#   GAZE_DEVICE    cuda:0 / cuda:1 ...
# ============================================================
DATA_DIR       = Path(os.environ.get(
    "GAZE_DATA_DIR", "/home/luxliang/datasets/EXPORT_PUPIL_ALL"))
IMG_SIZE       = 224        # 224匹配ResNet默认输入，略快于240
CLIP_LEN       = 4          # 4帧足够捕捉时序信息
FRAME_STRIDE   = 48         # 减少clip数量加速
HIDDEN_SIZE    = 256
LR             = 0.001
BATCH_SIZE     = int(os.environ.get("GAZE_BATCH", 16))  # 增大batch提高GPU利用率
TOTAL_EPOCHS   = int(os.environ.get("GAZE_EPOCHS", 15))
STAGE1_EPOCHS  = min(3, TOTAL_EPOCHS)
MAX_VIDEOS     = int(os.environ.get("GAZE_MAX_VIDEOS", 0))  # 0=全部
DEVICE_STR     = os.environ.get("GAZE_DEVICE", "cuda:0")
FPS            = 60.0 / FRAME_STRIDE * 4   # 实际等效帧率 ≈ 7.5fps

# 原有生物约束权重
W_C1_S1, W_C2_S1 = 1.0, 5.0
W_C1_S2, W_C2_S2 = 0.1, 0.5

# MS约束权重（相对MSE来说较小，避免主损失被压制）
W_MS_DURATION   = 0.001
W_MS_VPEAK      = 0.0001
W_MS_SATURATION = 0.01


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据加载
# ============================================================
def load_gaze_file(gaze_path):
    data = []
    with open(gaze_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(';')
            if len(parts) >= 4:
                try:
                    data.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except:
                    continue
    return np.array(data, dtype=np.float32)


def load_validity_file(validity_path):
    data = []
    with open(validity_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(';')
            if len(parts) >= 2:
                try:
                    data.append(int(parts[1]))
                except:
                    data.append(1)
    return np.array(data, dtype=np.int32)


class TEyeDDataset(Dataset):
    def __init__(self, sids, clip_len=CLIP_LEN, stride=FRAME_STRIDE,
                 img_size=IMG_SIZE, augment=False):
        self.clip_len = clip_len; self.stride = stride
        self.img_size = img_size; self.augment = augment; self.samples = []
        self.gaze_cache = {}   # sid -> np.ndarray
        self._vr_sid = None    # 当前缓存的视频对象
        self._vr = None

        for sid in sids:
            video_path    = DATA_DIR / f"{sid}.mp4pupil_seg_3D.mp4"
            gaze_path     = DATA_DIR / f"{sid}.mp4gaze_vec.txt"
            validity_path = DATA_DIR / f"{sid}.mp4validity_pupil.txt"
            if not (video_path.exists() and gaze_path.exists()):
                continue
            try:
                gaze_data = load_gaze_file(gaze_path)
                validity_data = (load_validity_file(validity_path)
                                 if validity_path.exists()
                                 else np.ones(len(gaze_data)))
                # 对齐长度
                n = min(len(gaze_data), len(validity_data))
                gaze_data = gaze_data[:n]
                validity_data = validity_data[:n]
                self.gaze_cache[sid] = gaze_data

                # 连续有效帧的片段（与成功脚本一致）
                need = (clip_len - 1) * stride + 1
                for start in range(0, n - need + 1, max(1, stride // 2)):
                    end = start + need
                    if np.all(validity_data[start:end] == 1):
                        self.samples.append({
                            'sid':       sid,
                            'start_idx': start,
                        })
            except Exception as e:
                print(f"  [skip] {sid}: {e}")
                continue

        self.samples.sort(key=lambda x: x['sid'])
        backend = "decord" if HAS_DECORD else "read_video"
        print(f"[TEyeD] {len(self.samples)} clips, {len(sids)} videos, backend={backend}")

    def __len__(self):
        return len(self.samples)

    def _get_vr(self, sid, video_path):
        """懒加载视频对象（decord VideoReader 或跳过）"""
        if sid != self._vr_sid:
            if HAS_DECORD:
                self._vr = DecordVR(str(video_path), ctx=decord_cpu(0))
            self._vr_sid = sid
        return self._vr

    def __getitem__(self, idx):
        sample  = self.samples[idx]
        sid     = sample['sid']
        video_path = DATA_DIR / f"{sid}.mp4pupil_seg_3D.mp4"
        start   = sample['start_idx']
        indices = [start + i * self.stride for i in range(self.clip_len)]

        if HAS_DECORD:
            vr = self._get_vr(sid, video_path)
            frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C] uint8 RGB
        else:
            # 备选：torchvision read_video（基于时间戳，帧精确）
            # 加载完整clip范围的所有帧，然后按stride抽取
            fps = 60.0
            s_pts = start / fps
            e_pts = (indices[-1] + 1) / fps
            all_frames, _, _ = read_video(str(video_path), start_pts=s_pts,
                                          end_pts=e_pts, pts_unit="sec")
            # 从start开始每stride帧取一帧（与成功脚本ablation_teyed_complete一致）
            rel = [i - start for i in indices]
            frames = all_frames[rel].numpy()

        # [T, H, W, C] uint8 -> [T, C, H, W] float32
        frames_np = frames.astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
        if frames_t.shape[2] != self.img_size or frames_t.shape[3] != self.img_size:
            frames_t = F.interpolate(frames_t, size=self.img_size,
                                     mode='bilinear', align_corners=False)

        # 目标：clip最后一帧的gaze方向
        target_frame = indices[-1]
        last_gaze = self.gaze_cache[sid][target_frame][:3].astype(np.float32)
        last_gaze = last_gaze / (np.linalg.norm(last_gaze) + 1e-8)

        return frames_t, torch.FloatTensor(last_gaze)


# ============================================================
# 模型定义
# ============================================================
class ResNet18GRUBio(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(512, HIDDEN_SIZE, 2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 3)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W).contiguous()
        f = self.backbone(x).flatten(1)
        f = self.feat_drop(f).reshape(B, T, -1)
        out, _ = self.gru(f)
        return self.head(out[:, -1]), out  # 返回原始logits + GRU序列（与成功脚本一致）


# ============================================================
# 原有生物约束损失（角度约束）
# ============================================================
def bio_angle_constraint(pred_gaze, stage=1):
    if stage == 1:
        w_c1, w_c2 = W_C1_S1, W_C2_S1
    else:
        w_c1, w_c2 = W_C1_S2, W_C2_S2
    loss = torch.tensor(0.0, device=pred_gaze.device)
    xy = torch.sqrt(pred_gaze[:, 0] ** 2 + pred_gaze[:, 1] ** 2 + 1e-8)
    z  = torch.abs(pred_gaze[:, 2]) + 1e-8
    angles = torch.atan2(xy, z)
    loss += w_c1 * torch.mean(torch.relu(angles - 40.0 * math.pi / 180.0))
    loss += w_c2 * torch.mean(torch.relu(angles - 35.0 * math.pi / 180.0))
    return loss


def compute_angular_error(pred, target):
    pred_n   = F.normalize(pred,   dim=-1)
    target_n = F.normalize(target, dim=-1)
    cos_sim  = F.cosine_similarity(pred_n, target_n, dim=-1).clamp(-0.9999, 0.9999)
    return torch.acos(cos_sim) * 180.0 / math.pi


# ============================================================
# 用MS约束从训练数据中自动估计群体参数（冷启动）
# ============================================================
def estimate_population_params_from_data(sids, max_videos=5):
    """
    直接使用群体先验初始化MS参数。
    （低帧率采样下saccade时长量化误差大，自动检测不可靠）
    """
    print(f"  [MS初始化] 使用群体先验: "
          f"k={POPULATION_PRIOR['k_i']:.2f}, a={POPULATION_PRIOR['a_i']:.2f}, "
          f"V0={POPULATION_PRIOR['V0_i']:.0f}, tau={POPULATION_PRIOR['tau_i']:.1f}")
    return POPULATION_PRIOR.copy()


# ============================================================
# 训练主函数
# ============================================================
def train_model(use_ms_constraint=True):
    seed_everything()
    device = torch.device(DEVICE_STR if torch.cuda.is_available() else 'cpu')

    # 获取所有视频SID
    all_sids = []
    for f in DATA_DIR.glob("*mp4gaze_vec.txt"):
        sid = f.name.replace('.mp4gaze_vec.txt', '')
        if (DATA_DIR / f"{sid}.mp4pupil_seg_3D.mp4").exists():
            all_sids.append(sid)
    all_sids.sort()
    if MAX_VIDEOS > 0:
        all_sids = all_sids[:MAX_VIDEOS]
        print(f"[烟囱测试] 仅使用前 {len(all_sids)} 个视频")
    n = len(all_sids)
    train_sids = all_sids[:int(n * 0.70)]
    val_sids   = all_sids[int(n * 0.70):int(n * 0.85)]
    test_sids  = all_sids[int(n * 0.85):]
    print(f"训练: {len(train_sids)}, 验证: {len(val_sids)}, 测试: {len(test_sids)}")

    # 数据集
    train_ds = TEyeDDataset(train_sids, augment=True)
    val_ds   = TEyeDDataset(val_sids)
    test_ds  = TEyeDDataset(test_sids)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 模型
    model = ResNet18GRUBio().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    # MS约束初始化
    ms_loss_fn = None
    if use_ms_constraint:
        init_params = estimate_population_params_from_data(train_sids)
        ms_loss_fn  = PersonalizedGazeLoss(
            fps=FPS,
            personalized_params=init_params,
            w_c1=W_C1_S2, w_c2=W_C2_S2,          # 角度约束权重（由ms_loss_fn内部处理）
            w_ms_duration=W_MS_DURATION,
            w_ms_vpeak=W_MS_VPEAK,
            w_ms_sat=W_MS_SATURATION,
        )

    tag = "full+MS" if use_ms_constraint else "full(无MS)"
    print(f"\nTEyeD - {tag}, stride={FRAME_STRIDE}, epochs={TOTAL_EPOCHS}")
    print(f"参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    best_val = 999
    save_name = 'ablation_teyed_full_ms_best.pt' if use_ms_constraint else 'ablation_teyed_full_best.pt'

    for epoch in range(TOTAL_EPOCHS):
        stage = 1 if epoch < STAGE1_EPOCHS else 2
        model.train()
        train_losses = []

        for frames, gazes in tqdm(train_loader, desc=f"Stage{stage} E{epoch+1}", leave=False):
            frames, gazes = frames.to(device), gazes.to(device)
            optimizer.zero_grad()

            # 前向传播（同时获取GRU序列用于MS约束）
            preds, gru_out = model(frames)

            # 主损失
            loss = F.mse_loss(preds, gazes)

            if use_ms_constraint:
                # ⚠ 已知局限（见 FINDINGS.md）：本 ablation 的采样几何下 MS 时序约束无效。
                #   1) clip 内 4 帧间隔 = FRAME_STRIDE/60 ≈ 0.8s，而 saccade 仅 20-80ms，
                #      完全落在采样间隙内 → 伪序列的"角速度/时长"是 0.8s 尺度的粗位移，
                #      并非 saccade 动态，MS 期望值无从匹配。
                #   2) pseudo_seq 在 no_grad 下计算 → ms 项对模型参数梯度恒为 0（no-op）。
                #   保留 no_grad 是刻意的：此采样率下启用梯度只会注入无意义信号。
                #   若要真正启用 MS 训练约束，需 ≥120fps 密集子窗口 + relative=True 归一化损失。
                with torch.no_grad():
                    pseudo_seq = model.head(gru_out)                    # [B, T, 3]
                    pseudo_seq = F.normalize(pseudo_seq, dim=-1)

                ms_result = ms_loss_fn(
                    pred_last=preds,
                    target=gazes,
                    gaze_seq=pseudo_seq,
                )
                # 实际生效的仅 bio 角度约束（依赖 preds，有梯度）；ms 项为常数(见上)
                loss = loss + ms_result['bio'] + ms_result['ms']
            else:
                # 原有角度约束
                loss = loss + bio_angle_constraint(preds, stage)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # 验证
        model.eval()
        val_errors = []
        with torch.no_grad():
            for frames, gazes in val_loader:
                frames, gazes = frames.to(device), gazes.to(device)
                preds, _ = model(frames)
                val_errors.extend(compute_angular_error(preds, gazes).cpu().numpy())

        val_mean = np.mean(val_errors) if val_errors else 999
        print(f"  Stage{stage} E{epoch+1}/{TOTAL_EPOCHS} - "
              f"Loss: {np.mean(train_losses):.4f}, Val: {val_mean:.3f} deg")

        if val_mean < best_val:
            best_val = val_mean
            torch.save(model.state_dict(), save_name)

    # 测试集评估
    model.load_state_dict(torch.load(save_name, weights_only=True))
    model.eval()
    test_errors = []
    with torch.no_grad():
        for frames, gazes in test_loader:
            frames, gazes = frames.to(device), gazes.to(device)
            preds, _ = model(frames)
            test_errors.extend(compute_angular_error(preds, gazes).cpu().numpy())

    mean_err   = np.mean(test_errors)
    median_err = np.median(test_errors)
    p90_err    = np.percentile(test_errors, 90)
    print(f"\n{'='*60}")
    print(f"[{tag}] 测试集结果:")
    print(f"  Mean={mean_err:.3f} deg, Median={median_err:.3f} deg, P90={p90_err:.3f} deg")
    print(f"{'='*60}\n")

    return {'mean': mean_err, 'median': median_err, 'p90': p90_err, 'tag': tag}


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    model_arg = sys.argv[1] if len(sys.argv) > 1 else 'full'

    if model_arg == 'full':
        result = train_model(use_ms_constraint=True)
    elif model_arg == 'full_no_ms':
        result = train_model(use_ms_constraint=False)
    elif model_arg == 'compare':
        # 对比实验：两个串行跑完
        r1 = train_model(use_ms_constraint=False)
        r2 = train_model(use_ms_constraint=True)
        print("\n对比结果:")
        print(f"  无MS约束:  Mean={r1['mean']:.3f}, Median={r1['median']:.3f}, P90={r1['p90']:.3f}")
        print(f"  有MS约束:  Mean={r2['mean']:.3f}, Median={r2['median']:.3f}, P90={r2['p90']:.3f}")
        improvement = r1['mean'] - r2['mean']
        print(f"  MS约束提升: {improvement:+.3f} deg ({'提升' if improvement > 0 else '下降'})")
    else:
        print(f"未知参数: {model_arg}")
        print("用法: python3 ablation_teyed_with_ms.py [full|full_no_ms|compare]")
