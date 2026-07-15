#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个性化 Main Sequence 核心模块 pytest 套件。

覆盖：
  - SaccadeDetector：幅度恢复精度 + 末尾眼跳flush回归
  - MainSequenceCalibrator：线性/指数拟合参数恢复
  - RLS / Bayesian 在线更新：收敛性 + RLS发散回归
  - MainSequenceConstraintLoss：生理序列损失 < 随机序列
  - MainSequenceVerifier：异常预测被纠正
  - UserMainSequenceBank：用户区分 + save/load 往返
"""
import numpy as np
import pytest
import torch

from personalized_main_sequence import (
    SaccadeDetector, MainSequenceCalibrator,
    RLSMainSequence, BayesianMainSequence,
    UserMainSequenceBank, MainSequenceConstraintLoss,
    MainSequenceVerifier, POPULATION_PRIOR,
)


# ---------------------------------------------------------------------------
# 合成数据工具
# ---------------------------------------------------------------------------
def _center_out_sequence(injected, T=300, out_frames=6, hold=10, ret_frames=30):
    """构造中心外跳+缓慢回中的注视序列。injected: [(start_frame, amp_deg), ...]"""
    center = np.array([0.0, 0.0, 1.0])
    gaze = np.tile(center, (T, 1)).astype(np.float64)
    for s, amp in injected:
        end = np.array([np.sin(np.radians(amp)), 0.0, np.cos(np.radians(amp))])
        for t in range(out_frames):
            a = t / (out_frames - 1)
            v = (1 - a) * center + a * end
            gaze[s + t] = v / np.linalg.norm(v)
        for t in range(out_frames, out_frames + hold):
            gaze[s + t] = end
        for t in range(ret_frames):
            a = t / (ret_frames - 1)
            v = (1 - a) * end + a * center
            gaze[s + out_frames + hold + t] = v / np.linalg.norm(v)
    return gaze


# ---------------------------------------------------------------------------
# SaccadeDetector
# ---------------------------------------------------------------------------
def test_detector_recovers_amplitudes():
    det = SaccadeDetector(fps=60.0)
    injected = [(40, 10), (140, 15), (230, 8)]
    gaze = _center_out_sequence(injected)
    found = det.detect(gaze)
    assert len(found) == len(injected), f"应检测到{len(injected)}个，实际{len(found)}"
    for (_, amp), s in zip(injected, found):
        assert abs(s['amplitude'] - amp) < 1.0, f"幅度误差过大: {s['amplitude']} vs {amp}"
        assert 20.0 <= s['duration'] <= 200.0


def test_detector_flushes_trailing_saccade():
    """回归：序列末尾仍在运动中的saccade必须被flush，不能静默丢弃。"""
    fps = 60.0
    center = np.array([0.0, 0.0, 1.0])
    T = 60
    gaze = np.tile(center, (T, 1)).astype(np.float64)
    # 前50帧静止，最后从第50帧开始一直运动到序列结束（无回落沿）
    end = np.array([np.sin(np.radians(12)), 0.0, np.cos(np.radians(12))])
    for i, t in enumerate(range(50, T)):
        a = i / (T - 50 - 1)
        v = (1 - a) * center + a * end
        gaze[t] = v / np.linalg.norm(v)
    found = det = SaccadeDetector(fps=fps).detect(gaze)
    assert len(found) == 1, "末尾进行中的saccade应被检测到"
    assert abs(found[0]['amplitude'] - 12.0) < 1.5


def test_detector_ignores_slow_drift():
    det = SaccadeDetector(fps=60.0)
    center = np.array([0.0, 0.0, 1.0])
    end = np.array([np.sin(np.radians(10)), 0.0, np.cos(np.radians(10))])
    T = 120
    gaze = np.zeros((T, 3))
    for t in range(T):                       # 10°用120帧漂移 → ~5°/s，远低于阈值
        a = t / (T - 1)
        v = (1 - a) * center + a * end
        gaze[t] = v / np.linalg.norm(v)
    assert det.detect(gaze) == []


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------
def test_calibrator_linear_recovers_params():
    rng = np.random.RandomState(0)
    amps = rng.uniform(3, 20, 60)
    a_true, k_true = 2.0, 2.5
    durs = a_true + k_true * amps + rng.randn(60) * 1.5
    p = MainSequenceCalibrator.fit_linear(amps, durs)
    assert abs(p['k_i'] - k_true) < 0.2
    assert abs(p['a_i'] - a_true) < 1.5      # 截距外推方差大，宽容差
    assert p['r_squared'] > 0.95


def test_calibrator_exponential_recovers_params():
    rng = np.random.RandomState(1)
    amps = rng.uniform(3, 25, 80)
    V0_true, tau_true = 580.0, 10.0
    vpeak = V0_true * (1 - np.exp(-amps / tau_true)) + rng.randn(80) * 15
    p = MainSequenceCalibrator.fit_exponential(amps, vpeak)
    assert abs(p['V0_i'] - V0_true) < 60
    assert abs(p['tau_i'] - tau_true) < 3.0


# ---------------------------------------------------------------------------
# 在线更新
# ---------------------------------------------------------------------------
def _run_updater(make, n=50, seed=0, a_true=2.0, k_true=2.5, noise=1.5):
    rng = np.random.RandomState(seed)
    u = make()
    out = None
    for _ in range(n):
        amp = rng.uniform(2, 20)
        dur = a_true + k_true * amp + rng.randn() * noise
        out = u.update(amp, dur)
    return abs(out['a_i'] - a_true) + abs(out['k_i'] - k_true)


def test_rls_converges_and_beats_prior():
    """回归：修复前 RLS 截距发散，最终误差比群体先验还差。"""
    prior_err = abs(POPULATION_PRIOR['a_i'] - 2.0) + abs(POPULATION_PRIOR['k_i'] - 2.5)
    errs = [_run_updater(RLSMainSequence, seed=s) for s in range(200)]
    mean_err = float(np.mean(errs))
    assert mean_err < 0.30, f"RLS平均误差 {mean_err:.3f} 过大（发散回归）"
    assert mean_err < prior_err, "RLS应优于群体先验"


def test_rls_matches_bayesian():
    rls = np.mean([_run_updater(RLSMainSequence, seed=s) for s in range(200)])
    bay = np.mean([_run_updater(BayesianMainSequence, seed=s) for s in range(200)])
    assert abs(rls - bay) < 0.05, f"RLS({rls:.3f}) 应与 Bayesian({bay:.3f}) 相当"


def test_rls_physiological_bounds():
    """极端观测下参数仍被约束在生理范围内。"""
    r = RLSMainSequence()
    for _ in range(30):
        r.update(amplitude=10.0, duration=999.0)   # 病态大值
    assert 0.0 <= r.theta[0] <= 10.0
    assert 1.0 <= r.theta[1] <= 5.0


def test_bayesian_anomaly_detect():
    p = MainSequenceCalibrator.fit_all([
        {'amplitude': a, 'duration': 2.0 + 2.5 * a, 'peak_velocity': 580 * (1 - np.exp(-a / 10))}
        for a in np.linspace(4, 20, 20)
    ])
    bay = BayesianMainSequence(a_prior=p['a_i'], k_prior=p['k_i'])
    assert bay.anomaly_detect(10.0, 2.0 + 2.5 * 10.0) is False   # 正常
    assert bay.anomaly_detect(10.0, 90.0) is True                # 太慢→异常


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------
def _make_saccade_batch(B=4, T=16, amp_deg=10.0):
    start = np.array([0.0, 0.0, 1.0])
    end = np.array([np.sin(np.radians(amp_deg)), 0.0, np.cos(np.radians(amp_deg))])
    seqs = []
    for _ in range(B):
        seq = []
        for t in range(T):
            a = t / (T - 1)
            a_s = 1 / (1 + np.exp(-10 * (a - 0.5)))
            v = (1 - a_s) * start + a_s * end
            seq.append(v / np.linalg.norm(v))
        seqs.append(seq)
    return torch.FloatTensor(np.array(seqs))


def test_loss_physiological_below_random():
    torch.manual_seed(0)
    params = {'a_i': 2.0, 'k_i': 2.5, 'V0_i': 580.0, 'tau_i': 10.0}
    loss_fn = MainSequenceConstraintLoss(fps=60.0, personalized_params=params)
    phys = loss_fn(_make_saccade_batch(amp_deg=10.0))
    rand = loss_fn(torch.nn.functional.normalize(torch.randn(4, 16, 3), dim=-1))
    assert phys['total'].item() < rand['total'].item()
    assert phys['total'].item() >= 0.0


def test_loss_short_sequence_safe():
    loss_fn = MainSequenceConstraintLoss(fps=60.0)
    out = loss_fn(torch.randn(2, 2, 3))          # T<3 边界
    assert out['total'].item() == 0.0


# ---------------------------------------------------------------------------
# 推理验证器
# ---------------------------------------------------------------------------
def test_verifier_corrects_anomaly():
    params = {'a_i': 2.0, 'k_i': 2.5, 'V0_i': 580.0, 'tau_i': 10.0}
    verifier = MainSequenceVerifier(personalized_params=params, fps=60.0)
    gaze0 = np.array([0.0, 0.0, 1.0])
    for _ in range(30):
        verifier.step(gaze0 + np.random.RandomState(0).randn(3) * 0.001)
    drift = np.array([np.sin(np.radians(15)), 0.0, np.cos(np.radians(15))])
    drift /= np.linalg.norm(drift)
    r = verifier.step(drift, prev_gaze=gaze0)
    before = np.degrees(np.arccos(np.clip(np.dot(drift, gaze0), -1, 1)))
    after = np.degrees(np.arccos(np.clip(np.dot(r['corrected'], gaze0), -1, 1)))
    assert after <= before + 1e-6                # 纠正不应放大偏差
    assert 0.0 <= r['confidence'] <= 1.0


# ---------------------------------------------------------------------------
# 用户库
# ---------------------------------------------------------------------------
def test_user_bank_distinguishes_users():
    rng = np.random.RandomState(3)
    bank = UserMainSequenceBank(updater_type='rls')
    slow = [{'amplitude': a, 'duration': 3.0 + 3.5 * a + rng.randn() * 1.0,
             'peak_velocity': 400 * (1 - np.exp(-a / 12))} for a in rng.uniform(3, 18, 30)]
    fast = [{'amplitude': a, 'duration': 1.5 + 1.8 * a + rng.randn() * 1.0,
             'peak_velocity': 700 * (1 - np.exp(-a / 8))} for a in rng.uniform(3, 18, 30)]
    bank.calibrate_user('slow', slow)
    bank.calibrate_user('fast', fast)
    assert bank.get_params('slow')['k_i'] > bank.get_params('fast')['k_i']


def test_user_bank_save_load_roundtrip(tmp_path):
    bank = UserMainSequenceBank(updater_type='bayesian')
    bank.update('u1', 10.0, 27.0)
    p_before = bank.get_params('u1')
    path = str(tmp_path / 'bank.pkl')
    bank.save(path)
    bank2 = UserMainSequenceBank()
    bank2.load(path)
    p_after = bank2.get_params('u1')
    assert abs(p_before['a_i'] - p_after['a_i']) < 1e-9
    assert abs(p_before['k_i'] - p_after['k_i']) < 1e-9


def test_unknown_user_returns_prior():
    bank = UserMainSequenceBank()
    p = bank.get_params('never_seen')
    assert p['a_i'] == POPULATION_PRIOR['a_i']
    assert p['k_i'] == POPULATION_PRIOR['k_i']
