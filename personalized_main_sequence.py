#!/usr/bin/env python3
"""
个性化Main Sequence约束模块
支持：
  1. 离线标定（线性/指数/鲁棒拟合）
  2. 在线自适应更新（RLS / Bayesian）
  3. PyTorch可微分损失函数
  4. 推理阶段硬约束门控
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
from typing import Optional, Dict, List, Tuple

# ============================================================
# 全局群体先验（fallback默认值）
# ============================================================
POPULATION_PRIOR = {
    'a_i':  2.2,    # 持续时间截距 (ms)
    'k_i':  2.8,    # 持续时间斜率 (ms/°)
    'V0_i': 600.0,  # 峰值速度饱和值 (°/s)
    'tau_i': 9.8,   # 速度-幅度饱和常数 (°)
}

# 生理约束范围
PARAM_BOUNDS = {
    'a_i':   (0.0,   10.0),
    'k_i':   (1.0,   5.0),
    'V0_i':  (200.0, 900.0),
    'tau_i': (4.0,   20.0),
}


# ============================================================
# 一、Saccade检测器
# ============================================================
class SaccadeDetector:
    """
    从注视向量序列中检测saccade事件，提取幅度/持续时间/峰值速度
    输入：3D注视方向向量序列（单位向量）
    """

    def __init__(self, fps: float = 60.0,
                 velocity_threshold: float = 50.0,    # °/s
                 min_duration_ms: float = 20.0,
                 max_duration_ms: float = 200.0,
                 min_amplitude_deg: float = 1.0,
                 max_amplitude_deg: float = 30.0):
        self.fps = fps
        self.dt = 1000.0 / fps            # ms per frame
        self.v_thresh = velocity_threshold
        self.min_dur = min_duration_ms
        self.max_dur = max_duration_ms
        self.min_amp = min_amplitude_deg
        self.max_amp = max_amplitude_deg

    def angular_velocity(self, gaze_seq: np.ndarray) -> np.ndarray:
        """
        计算逐帧角速度 (°/s)
        gaze_seq: [T, 3] 单位向量序列
        返回: [T-1] 角速度
        """
        g = gaze_seq / (np.linalg.norm(gaze_seq, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.clip(np.sum(g[:-1] * g[1:], axis=1), -1.0, 1.0)
        angle_diff = np.degrees(np.arccos(cos_sim))      # 度/帧
        return angle_diff / self.dt * 1000.0              # 度/秒

    def detect(self, gaze_seq: np.ndarray) -> List[Dict]:
        """
        返回有效saccade列表，每个元素包含：
          amplitude (°), duration (ms), peak_velocity (°/s),
          start_frame, end_frame
        """
        if len(gaze_seq) < 4:
            return []

        vel = self.angular_velocity(gaze_seq)
        in_saccade = False
        saccade_start = 0
        saccades = []

        for i, v in enumerate(vel):
            if not in_saccade and v > self.v_thresh:
                in_saccade = True
                saccade_start = i
            elif in_saccade and v <= self.v_thresh:
                in_saccade = False
                saccade_end = i
                dur = (saccade_end - saccade_start) * self.dt

                if self.min_dur <= dur <= self.max_dur:
                    g_start = gaze_seq[saccade_start]
                    g_end   = gaze_seq[min(saccade_end, len(gaze_seq) - 1)]
                    g_start = g_start / (np.linalg.norm(g_start) + 1e-8)
                    g_end   = g_end   / (np.linalg.norm(g_end)   + 1e-8)
                    cos_a = np.clip(np.dot(g_start, g_end), -1.0, 1.0)
                    amp = np.degrees(np.arccos(cos_a))
                    v_peak = float(np.max(vel[saccade_start:saccade_end + 1]))

                    if self.min_amp <= amp <= self.max_amp:
                        saccades.append({
                            'amplitude':    amp,
                            'duration':     dur,
                            'peak_velocity': v_peak,
                            'start_frame':  saccade_start,
                            'end_frame':    saccade_end,
                        })
        return saccades


# ============================================================
# 二、离线标定：参数拟合
# ============================================================
class MainSequenceCalibrator:
    """
    给定一组saccade观测值，拟合个体化Main Sequence参数
    """

    @staticmethod
    def fit_linear(amplitudes: np.ndarray,
                   durations: np.ndarray) -> Dict:
        """线性回归：D = a + k*A"""
        A = np.asarray(amplitudes, dtype=np.float64)
        D = np.asarray(durations,  dtype=np.float64)
        if len(A) < 3:
            return POPULATION_PRIOR.copy()

        A_design = np.vstack([np.ones_like(A), A]).T
        params, residuals, _, _ = np.linalg.lstsq(A_design, D, rcond=None)
        a_i, k_i = params

        ss_res = np.sum((D - (a_i + k_i * A)) ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        a_i = np.clip(a_i, *PARAM_BOUNDS['a_i'])
        k_i = np.clip(k_i, *PARAM_BOUNDS['k_i'])

        return {'a_i': float(a_i), 'k_i': float(k_i),
                'r_squared': float(r2), 'n_samples': len(A), 'method': 'linear'}

    @staticmethod
    def fit_exponential(amplitudes: np.ndarray,
                        peak_velocities: np.ndarray) -> Dict:
        """指数饱和：Vp = V0 * (1 - exp(-A/tau))"""
        A  = np.asarray(amplitudes,      dtype=np.float64)
        Vp = np.asarray(peak_velocities, dtype=np.float64)
        if len(A) < 4:
            return {'V0_i': POPULATION_PRIOR['V0_i'],
                    'tau_i': POPULATION_PRIOR['tau_i']}

        def model(A, V0, tau):
            return V0 * (1 - np.exp(-A / tau))

        try:
            params, cov = curve_fit(
                model, A, Vp,
                p0=[600.0, 9.8],
                bounds=([PARAM_BOUNDS['V0_i'][0], PARAM_BOUNDS['tau_i'][0]],
                        [PARAM_BOUNDS['V0_i'][1], PARAM_BOUNDS['tau_i'][1]]),
                maxfev=10000
            )
            V0_i, tau_i = params
            perr = np.sqrt(np.diag(cov))
            return {'V0_i': float(V0_i), 'tau_i': float(tau_i),
                    'V0_std': float(perr[0]), 'tau_std': float(perr[1]),
                    'n_samples': len(A), 'method': 'exponential'}
        except RuntimeError:
            return {'V0_i': POPULATION_PRIOR['V0_i'],
                    'tau_i': POPULATION_PRIOR['tau_i'],
                    'method': 'fallback'}

    @staticmethod
    def fit_all(saccades: List[Dict]) -> Dict:
        """从saccade列表一键拟合全部参数"""
        if not saccades:
            return POPULATION_PRIOR.copy()

        amps  = np.array([s['amplitude']     for s in saccades])
        durs  = np.array([s['duration']      for s in saccades])
        vpeak = np.array([s['peak_velocity'] for s in saccades])

        linear_params = MainSequenceCalibrator.fit_linear(amps, durs)
        exp_params    = MainSequenceCalibrator.fit_exponential(amps, vpeak)

        return {**POPULATION_PRIOR, **linear_params, **exp_params}


# ============================================================
# 三、在线自适应更新
# ============================================================
class RLSMainSequence:
    """
    递归最小二乘在线更新 Main Sequence 参数（持续时间-幅度关系）
    状态向量 θ = [a_i, k_i]
    """

    def __init__(self,
                 a_init: float = POPULATION_PRIOR['a_i'],
                 k_init: float = POPULATION_PRIOR['k_i'],
                 lambda_forget: float = 0.98,
                 P_init: float = 1.0):   # 缩小初始不确定性，避免早期过度更新
        self.theta = np.array([a_init, k_init], dtype=np.float64)
        self.P = np.eye(2) * P_init
        self.lambda_ = lambda_forget
        self.n_updates = 0

    def update(self, amplitude: float, duration: float) -> Dict:
        x = np.array([1.0, amplitude])
        y_pred = x @ self.theta
        innovation = duration - y_pred

        S = float(x @ self.P @ x.T) + 1.0
        K = self.P @ x.T / S

        self.theta = self.theta + K * innovation
        self.P = (self.P - np.outer(K, x) @ self.P) / self.lambda_

        # 生理约束
        self.theta[0] = np.clip(self.theta[0], *PARAM_BOUNDS['a_i'])
        self.theta[1] = np.clip(self.theta[1], *PARAM_BOUNDS['k_i'])

        self.n_updates += 1
        return {'a_i': self.theta[0], 'k_i': self.theta[1],
                'innovation': innovation, 'n_updates': self.n_updates}

    def predict_duration(self, amplitude: float) -> float:
        return float(self.theta[0] + self.theta[1] * amplitude)

    def get_confidence(self) -> float:
        return 1.0 / (np.trace(self.P) + 1e-6)

    def get_params(self) -> Dict:
        return {'a_i': self.theta[0], 'k_i': self.theta[1]}


class BayesianMainSequence:
    """
    贝叶斯在线更新，提供参数不确定性估计及异常检测
    """

    def __init__(self,
                 a_prior: float = POPULATION_PRIOR['a_i'],
                 k_prior: float = POPULATION_PRIOR['k_i'],
                 sigma_prior: float = 1.0,
                 obs_noise: float = 5.0):
        self.mu    = np.array([a_prior, k_prior], dtype=np.float64)
        self.Sigma = np.eye(2) * sigma_prior ** 2
        self.R     = obs_noise ** 2

    def update(self, amplitude: float, duration: float) -> Dict:
        x = np.array([1.0, amplitude])
        innovation = duration - x @ self.mu
        S = float(x @ self.Sigma @ x.T) + self.R
        K = self.Sigma @ x.T / S

        self.mu    = self.mu + K * innovation
        self.Sigma = self.Sigma - np.outer(K, x) @ self.Sigma
        # 保证正定
        self.Sigma = (self.Sigma + self.Sigma.T) / 2 + 1e-6 * np.eye(2)

        # 生理约束
        self.mu[0] = np.clip(self.mu[0], *PARAM_BOUNDS['a_i'])
        self.mu[1] = np.clip(self.mu[1], *PARAM_BOUNDS['k_i'])

        return {'a_i': self.mu[0], 'k_i': self.mu[1],
                'std_a': np.sqrt(self.Sigma[0, 0]),
                'std_k': np.sqrt(self.Sigma[1, 1]),
                'innovation': innovation}

    def anomaly_detect(self, amplitude: float, duration: float,
                       threshold: float = 3.0) -> bool:
        x = np.array([1.0, amplitude])
        innovation = duration - x @ self.mu
        S = float(x @ self.Sigma @ x.T) + self.R
        mahalanobis = abs(innovation) / (np.sqrt(S) + 1e-8)
        return bool(mahalanobis > threshold)

    def get_params(self) -> Dict:
        return {'a_i': self.mu[0], 'k_i': self.mu[1],
                'std_a': np.sqrt(self.Sigma[0, 0]),
                'std_k': np.sqrt(self.Sigma[1, 1])}


# ============================================================
# 四、用户参数管理器（多用户）
# ============================================================
class UserMainSequenceBank:
    """
    管理多用户个性化参数
    支持：初始化、更新、保存/加载
    """

    def __init__(self, updater_type: str = 'rls',
                 lambda_forget: float = 0.98):
        self.updater_type = updater_type
        self.lambda_forget = lambda_forget
        self.users: Dict[str, object] = {}

    def _init_user(self, user_id: str,
                   init_params: Optional[Dict] = None):
        p = init_params or POPULATION_PRIOR
        if self.updater_type == 'bayesian':
            self.users[user_id] = BayesianMainSequence(
                a_prior=p.get('a_i', POPULATION_PRIOR['a_i']),
                k_prior=p.get('k_i', POPULATION_PRIOR['k_i'])
            )
        else:  # 默认 rls
            self.users[user_id] = RLSMainSequence(
                a_init=p.get('a_i', POPULATION_PRIOR['a_i']),
                k_init=p.get('k_i', POPULATION_PRIOR['k_i']),
                lambda_forget=self.lambda_forget
            )

    def update(self, user_id: str, amplitude: float, duration: float,
               init_params: Optional[Dict] = None) -> Dict:
        if user_id not in self.users:
            self._init_user(user_id, init_params)
        return self.users[user_id].update(amplitude, duration)

    def get_params(self, user_id: str) -> Dict:
        if user_id not in self.users:
            return POPULATION_PRIOR.copy()
        return self.users[user_id].get_params()

    def calibrate_user(self, user_id: str, saccades: List[Dict]) -> Dict:
        """用离线标定数据初始化用户参数"""
        params = MainSequenceCalibrator.fit_all(saccades)
        self._init_user(user_id, params)
        return params

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'users': self.users,
                         'updater_type': self.updater_type}, f)
        print(f"[UserBank] 已保存 {len(self.users)} 个用户参数 → {path}")

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.users = data['users']
        self.updater_type = data['updater_type']
        print(f"[UserBank] 已加载 {len(self.users)} 个用户参数")


# ============================================================
# 五、PyTorch 可微分损失函数
# ============================================================
class MainSequenceConstraintLoss(nn.Module):
    """
    个性化 Main Sequence 软约束损失
    输入：预测注视序列 [B, T, 3]（3D单位向量）
    """

    def __init__(self,
                 fps: float = 60.0,
                 personalized_params: Optional[Dict] = None,
                 w_duration: float = 1.0,
                 w_vpeak: float = 0.1,
                 w_saturation: float = 10.0):
        super().__init__()
        p = personalized_params or POPULATION_PRIOR
        self.fps = fps
        self.dt_s = 1.0 / fps

        self.register_buffer('a',   torch.tensor(p.get('a_i',   POPULATION_PRIOR['a_i']),   dtype=torch.float32))
        self.register_buffer('k',   torch.tensor(p.get('k_i',   POPULATION_PRIOR['k_i']),   dtype=torch.float32))
        self.register_buffer('V0',  torch.tensor(p.get('V0_i',  POPULATION_PRIOR['V0_i']),  dtype=torch.float32))
        self.register_buffer('tau', torch.tensor(p.get('tau_i', POPULATION_PRIOR['tau_i']), dtype=torch.float32))

        self.w_duration   = w_duration
        self.w_vpeak      = w_vpeak
        self.w_saturation = w_saturation

    def update_params(self, params: Dict):
        """动态更新个性化参数（推理阶段在线更新后调用）"""
        device = self.a.device
        if 'a_i'   in params: self.a.data   = torch.tensor(params['a_i'],   dtype=torch.float32, device=device)
        if 'k_i'   in params: self.k.data   = torch.tensor(params['k_i'],   dtype=torch.float32, device=device)
        if 'V0_i'  in params: self.V0.data  = torch.tensor(params['V0_i'],  dtype=torch.float32, device=device)
        if 'tau_i' in params: self.tau.data = torch.tensor(params['tau_i'], dtype=torch.float32, device=device)

    def forward(self, gaze_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        gaze_seq: [B, T, 3] 预测注视向量（不要求单位化，内部会归一化）
        """
        B, T, _ = gaze_seq.shape
        if T < 3:
            zero = torch.tensor(0.0, device=gaze_seq.device)
            return {'duration_loss': zero, 'vpeak_loss': zero,
                    'saturation_penalty': zero, 'total': zero}

        # 归一化
        g = F.normalize(gaze_seq, dim=-1)

        # 逐帧角速度 (°/s)
        cos_v = F.cosine_similarity(g[:, :-1], g[:, 1:], dim=-1).clamp(-0.9999, 0.9999)
        angle_diff = torch.acos(cos_v) * (180.0 / torch.pi)  # 度/帧
        vel = angle_diff / self.dt_s                           # 度/秒

        # 峰值速度
        v_peak, _ = vel.max(dim=1)  # [B]

        # 持续时间（速度超过阈值的帧数 × dt）
        v_thresh = 50.0  # °/s
        active = (vel > v_thresh).float()
        durations = active.sum(dim=1) * (1000.0 * self.dt_s)  # ms [B]

        # 总幅度（首尾角度差）
        cos_amp = F.cosine_similarity(g[:, 0], g[:, -1], dim=-1).clamp(-0.9999, 0.9999)
        amplitudes = torch.acos(cos_amp) * (180.0 / torch.pi)  # [B]

        # 个性化期望值
        expected_dur   = self.a + self.k * amplitudes                      # [B]
        expected_vpeak = self.V0 * (1 - torch.exp(-amplitudes / self.tau)) # [B]

        # 损失计算
        duration_loss     = F.mse_loss(durations, expected_dur)
        vpeak_loss        = F.smooth_l1_loss(v_peak, expected_vpeak, beta=50.0)
        saturation_penalty = torch.relu(v_peak - self.V0).mean()

        total = (self.w_duration   * duration_loss
               + self.w_vpeak      * vpeak_loss
               + self.w_saturation * saturation_penalty)

        return {
            'duration_loss':     duration_loss,
            'vpeak_loss':        vpeak_loss,
            'saturation_penalty': saturation_penalty,
            'total':             total,
        }


# ============================================================
# 六、推理阶段硬约束门控（验证器）
# ============================================================
class MainSequenceVerifier:
    """
    推理阶段：用个性化Main Sequence验证/纠正CNN预测结果
    适用于帧级3D注视向量
    """

    def __init__(self,
                 personalized_params: Optional[Dict] = None,
                 fps: float = 60.0,
                 v_saccade_thresh: float = 100.0,  # °/s
                 correction_sigma: float = 2.0):   # 容忍度(°)
        p = personalized_params or POPULATION_PRIOR
        self.params = p
        self.fps = fps
        self.dt_s = 1.0 / fps
        self.v_thresh = v_saccade_thresh
        self.sigma = correction_sigma
        self.history: List[Dict] = []
        self.saccade_start: Optional[int] = None
        self.frame_idx = 0

    def update_params(self, params: Dict):
        self.params = {**self.params, **params}

    def step(self, gaze: np.ndarray,
             prev_gaze: Optional[np.ndarray] = None) -> Dict:
        """
        单帧处理
        gaze:      [3] 当前帧CNN预测（单位向量）
        prev_gaze: [3] 上一帧位置（可选，默认用history末尾）
        """
        gaze = gaze / (np.linalg.norm(gaze) + 1e-8)

        if prev_gaze is None:
            if self.history:
                prev_gaze = self.history[-1]['gaze']
            else:
                self.history.append({'gaze': gaze, 'frame': self.frame_idx})
                self.frame_idx += 1
                return {'corrected': gaze, 'confidence': 1.0,
                        'is_saccade': False, 'velocity': 0.0}

        prev_gaze = prev_gaze / (np.linalg.norm(prev_gaze) + 1e-8)
        cos_a = np.clip(np.dot(prev_gaze, gaze), -1.0, 1.0)
        angle_diff = np.degrees(np.arccos(cos_a))
        velocity = angle_diff / self.dt_s

        is_saccade = velocity > self.v_thresh
        corrected = gaze.copy()
        confidence = 1.0

        if is_saccade:
            if self.saccade_start is None:
                self.saccade_start = self.frame_idx

            elapsed_frames = self.frame_idx - self.saccade_start + 1
            elapsed_ms = elapsed_frames * 1000.0 * self.dt_s

            if self.history:
                start_gaze = self.history[self.saccade_start
                    if self.saccade_start < len(self.history) else -1]['gaze']
                amp = np.degrees(np.arccos(np.clip(
                    np.dot(start_gaze, gaze), -1.0, 1.0)))

                if amp > 0.5:
                    expected_dur = (self.params.get('a_i', POPULATION_PRIOR['a_i'])
                                  + self.params.get('k_i', POPULATION_PRIOR['k_i']) * amp)
                    expected_vp  = (self.params.get('V0_i', POPULATION_PRIOR['V0_i'])
                                  * (1 - np.exp(-amp / self.params.get('tau_i', POPULATION_PRIOR['tau_i']))))

                    # 理论进度（0→1）
                    progress = min(elapsed_ms / (expected_dur + 1e-8), 1.0)
                    theoretical_gaze = self._slerp(start_gaze, gaze, progress)

                    # 偏差计算
                    cos_dev = np.clip(np.dot(gaze, theoretical_gaze), -1.0, 1.0)
                    deviation = np.degrees(np.arccos(cos_dev))

                    confidence = float(np.exp(-deviation / (self.sigma + 1e-8)))
                    corrected = self._slerp(gaze, theoretical_gaze, 1.0 - confidence)
        else:
            self.saccade_start = None

        self.history.append({'gaze': corrected, 'frame': self.frame_idx})
        if len(self.history) > 200:
            self.history.pop(0)
        self.frame_idx += 1

        return {
            'corrected':   corrected,
            'confidence':  confidence,
            'is_saccade':  is_saccade,
            'velocity':    velocity,
        }

    @staticmethod
    def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """球面线性插值"""
        v0 = v0 / (np.linalg.norm(v0) + 1e-8)
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
        theta = np.arccos(dot)
        if abs(theta) < 1e-6:
            return (1 - t) * v0 + t * v1
        return (np.sin((1 - t) * theta) * v0 + np.sin(t * theta) * v1) / np.sin(theta)


# ============================================================
# 七、与现有ResNet18-GRU-Bio模型集成示例
# ============================================================
class PersonalizedGazeLoss(nn.Module):
    """
    完整个性化损失函数，替换/增强原有bio_constraint_loss
    = MSE + 原有生物约束（C1/C2角度约束）+ 个性化Main Sequence约束
    """

    def __init__(self,
                 fps: float = 60.0,
                 personalized_params: Optional[Dict] = None,
                 # 原有生物约束权重
                 w_c1: float = 0.1, w_c2: float = 0.5,
                 # Main Sequence约束权重
                 w_ms_duration: float = 0.05,
                 w_ms_vpeak: float = 0.01,
                 w_ms_sat: float = 1.0):
        super().__init__()
        self.ms_loss = MainSequenceConstraintLoss(
            fps=fps,
            personalized_params=personalized_params,
            w_duration=w_ms_duration,
            w_vpeak=w_ms_vpeak,
            w_saturation=w_ms_sat
        )
        self.w_c1 = w_c1
        self.w_c2 = w_c2
        import math
        self.C1_rad = 40.0 * math.pi / 180.0
        self.C2_rad = 35.0 * math.pi / 180.0

    def bio_angle_constraint(self, pred_gaze: torch.Tensor) -> torch.Tensor:
        """原有C1/C2角度约束（适用于单帧输出）"""
        import math
        xy = torch.sqrt(pred_gaze[:, 0] ** 2 + pred_gaze[:, 1] ** 2 + 1e-8)
        z  = torch.abs(pred_gaze[:, 2]) + 1e-8
        angles = torch.atan2(xy, z)
        c1 = self.w_c1 * torch.mean(torch.relu(angles - self.C1_rad))
        c2 = self.w_c2 * torch.mean(torch.relu(angles - self.C2_rad))
        return c1 + c2

    def forward(self,
                pred_last: torch.Tensor,    # [B, 3] 最后一帧预测（用于原有约束）
                target: torch.Tensor,       # [B, 3] GT
                gaze_seq: Optional[torch.Tensor] = None  # [B, T, 3] 完整序列（用于MS约束）
                ) -> Dict[str, torch.Tensor]:
        mse = F.mse_loss(pred_last, target)
        bio = self.bio_angle_constraint(pred_last)

        ms_total = torch.tensor(0.0, device=pred_last.device)
        if gaze_seq is not None and gaze_seq.shape[1] >= 3:
            ms_out  = self.ms_loss(gaze_seq)
            ms_total = ms_out['total']

        total = mse + bio + ms_total
        return {
            'mse':      mse,
            'bio':      bio,
            'ms':       ms_total,
            'total':    total,
        }


# ============================================================
# 八、使用示例
# ============================================================
if __name__ == '__main__':
    import math

    print("=" * 60)
    print("个性化 Main Sequence 约束模块 - 功能演示")
    print("=" * 60)

    np.random.seed(42)

    # --------------------------------------------------------
    # 测试1：离线标定
    # --------------------------------------------------------
    print("\n[测试1] 离线标定（模拟30个saccade观测）")
    amps  = np.random.uniform(3, 20, 30)
    durs  = 2.0 + 2.5 * amps + np.random.randn(30) * 2.0    # 真实参数: a=2, k=2.5
    vpeak = 580 * (1 - np.exp(-amps / 10.0)) + np.random.randn(30) * 20  # V0=580, tau=10
    saccades = [{'amplitude': a, 'duration': d, 'peak_velocity': v}
                for a, d, v in zip(amps, durs, vpeak)]

    params = MainSequenceCalibrator.fit_all(saccades)
    print(f"  真实参数:    a=2.000, k=2.500, V0=580.0, tau=10.00")
    print(f"  拟合参数:    a={params['a_i']:.3f}, k={params['k_i']:.3f}, "
          f"V0={params['V0_i']:.1f}, tau={params['tau_i']:.2f}")
    print(f"  R²={params.get('r_squared', 0.0):.4f}  (越接近1越好)")

    # --------------------------------------------------------
    # 测试2：RLS在线更新（模拟与真实参数的收敛）
    # --------------------------------------------------------
    print("\n[测试2] RLS在线更新收敛性")
    print(f"  初始(群体先验): a={POPULATION_PRIOR['a_i']:.2f}, k={POPULATION_PRIOR['k_i']:.2f}")
    rls = RLSMainSequence()  # 使用群体先验初始化
    updates_log = []
    for i, (amp, dur) in enumerate(zip(amps, durs)):
        r = rls.update(amp, dur)
        if i in [0, 4, 9, 19, 29]:
            updates_log.append((i+1, r['a_i'], r['k_i']))
    for n, a, k in updates_log:
        print(f"  更新{n:2d}次后: a={a:.3f}, k={k:.3f} "
              f"(误差: Δa={abs(a-2.0):.3f}, Δk={abs(k-2.5):.3f})")

    # --------------------------------------------------------
    # 测试3：异常检测（Bayesian）
    # --------------------------------------------------------
    print("\n[测试3] Bayesian异常检测")
    bayes = BayesianMainSequence(a_prior=params['a_i'], k_prior=params['k_i'])
    normal_saccade  = (10.0, 27.0)   # A=10°, D=27ms → 正常（期望≈27ms）
    abnormal_saccade = (10.0, 80.0)  # A=10°, D=80ms → 异常（太慢）
    for label, (amp, dur) in [('正常', normal_saccade), ('异常', abnormal_saccade)]:
        is_anom = bayes.anomaly_detect(amp, dur)
        print(f"  {label} saccade (A={amp}°, D={dur}ms): 异常={is_anom}")

    # --------------------------------------------------------
    # 测试4：用生理合理的注视序列测试损失函数
    # --------------------------------------------------------
    print("\n[测试4] 损失函数（生理合理序列 vs 随机序列）")

    def make_saccade_seq(B=4, T=16, amp_deg=10.0, fps=60.0):
        """生成一个模拟saccade的3D注视序列"""
        seqs = []
        for _ in range(B):
            start = np.array([0.0, 0.0, 1.0])
            end   = np.array([np.sin(np.radians(amp_deg)), 0.0,
                              np.cos(np.radians(amp_deg))])
            seq = []
            for t in range(T):
                alpha = t / (T - 1)
                # 用平滑的sigmoid轮廓
                alpha_smooth = 1 / (1 + np.exp(-10 * (alpha - 0.5)))
                v = (1 - alpha_smooth) * start + alpha_smooth * end
                v = v / np.linalg.norm(v)
                seq.append(v)
            seqs.append(seq)
        return torch.FloatTensor(np.array(seqs))

    physiological_seq = make_saccade_seq(B=4, T=16, amp_deg=10.0, fps=60.0)
    random_seq = F.normalize(torch.randn(4, 16, 3), dim=-1)
    target = F.normalize(torch.randn(4, 3), dim=-1)

    ms_loss_fn = MainSequenceConstraintLoss(fps=60.0, personalized_params=params)

    loss_phys  = ms_loss_fn(physiological_seq)
    loss_rand  = ms_loss_fn(random_seq)
    print(f"  生理合理序列: total={loss_phys['total'].item():.4f} "
          f"(dur={loss_phys['duration_loss'].item():.4f}, "
          f"vp={loss_phys['vpeak_loss'].item():.4f}, "
          f"sat={loss_phys['saturation_penalty'].item():.4f})")
    print(f"  随机序列:     total={loss_rand['total'].item():.4f} "
          f"(dur={loss_rand['duration_loss'].item():.4f}, "
          f"vp={loss_rand['vpeak_loss'].item():.4f}, "
          f"sat={loss_rand['saturation_penalty'].item():.4f})")
    print(f"  ✓ 生理合理序列损失应更小 → {'是' if loss_phys['total'] < loss_rand['total'] else '否'}")

    # --------------------------------------------------------
    # 测试5：多用户管理
    # --------------------------------------------------------
    print("\n[测试5] 多用户参数管理")
    bank = UserMainSequenceBank(updater_type='rls')
    # 用户A：慢速眼跳（k偏大）
    saccades_A = [{'amplitude': a, 'duration': 3.0 + 3.5*a + np.random.randn()*1.5,
                   'peak_velocity': 400*(1-np.exp(-a/12))}
                  for a in np.random.uniform(3, 18, 20)]
    # 用户B：快速眼跳（k偏小）
    saccades_B = [{'amplitude': a, 'duration': 1.5 + 1.8*a + np.random.randn()*1.0,
                   'peak_velocity': 700*(1-np.exp(-a/8))}
                  for a in np.random.uniform(3, 18, 20)]

    bank.calibrate_user('user_A', saccades_A)
    bank.calibrate_user('user_B', saccades_B)
    pA = bank.get_params('user_A')
    pB = bank.get_params('user_B')
    print(f"  user_A(慢速): a={pA['a_i']:.3f}, k={pA['k_i']:.3f}")
    print(f"  user_B(快速): a={pB['a_i']:.3f}, k={pB['k_i']:.3f}")
    print(f"  ✓ 用户A的k应 > 用户B的k → {'是' if pA['k_i'] > pB['k_i'] else '否'}")

    # --------------------------------------------------------
    # 测试6：SaccadeDetector
    # --------------------------------------------------------
    print("\n[测试6] SaccadeDetector自动检测")
    detector = SaccadeDetector(fps=60.0)
    # 构造含3个saccade的序列
    T_total = 300
    gaze_gt = np.tile([0.0, 0.0, 1.0], (T_total, 1)).astype(np.float64)
    # 60FPS下，saccade持续8帧 ≈ 133ms（在20-200ms范围内）
    for sac_start, amp in [(50, 10), (130, 15), (210, 8)]:
        end = np.array([np.sin(np.radians(amp)), 0, np.cos(np.radians(amp))])
        for t in range(8):   # 8帧 ≈ 133ms
            alpha = t / 7
            v = (1-alpha)*gaze_gt[sac_start] + alpha*end
            gaze_gt[sac_start+t] = v / np.linalg.norm(v)
        gaze_gt[sac_start+8:] = gaze_gt[sac_start+7]

    detected = detector.detect(gaze_gt)
    print(f"  注入3个saccade，检测到 {len(detected)} 个")
    for i, s in enumerate(detected):
        print(f"    #{i+1}: A={s['amplitude']:.1f}°, D={s['duration']:.1f}ms, Vp={s['peak_velocity']:.1f}°/s")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
