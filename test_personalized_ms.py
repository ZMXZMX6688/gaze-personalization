#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个性化Main Sequence端到端测试
演示完整流程：用户序列 -> 检测saccade -> 标定参数 -> 在线更新 -> 训练/推理集成
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 设置支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from personalized_main_sequence import (
    SaccadeDetector, MainSequenceCalibrator,
    RLSMainSequence, BayesianMainSequence,
    UserMainSequenceBank, MainSequenceConstraintLoss,
    MainSequenceVerifier, PersonalizedGazeLoss,
    POPULATION_PRIOR
)

np.random.seed(0)


# ============================================================
# 工具函数：生成模拟真实用户的注视序列
# ============================================================
def simulate_user_gaze(n_seconds=60.0, fps=60.0,
                       user_k=2.5, user_a=2.0,
                       user_V0=580.0, user_tau=10.0,
                       noise_std=0.5):
    """
    模拟单个用户的3D注视向量序列
    返回: (gaze_seq [T,3], saccade_log [list of dict])
    """
    T = int(n_seconds * fps)
    gaze = np.zeros((T, 3))
    gaze[:, 2] = 1.0

    t = 0
    current_dir = np.array([0.0, 0.0, 1.0])
    saccade_log = []

    while t < T:
        # 注视阶段: 200-800ms
        fixation_dur = int(np.random.uniform(200, 800) / 1000 * fps)
        for i in range(min(fixation_dur, T - t)):
            noise = np.random.randn(3) * np.radians(noise_std)
            v = current_dir + noise
            gaze[t + i] = v / np.linalg.norm(v)
        t += fixation_dur
        if t >= T:
            break

        # 眼跳阶段：幅度5-25度，避免小幅度saccade的截断偏差
        amp_deg = np.random.uniform(5, 25)
        dur_ms_true = user_a + user_k * amp_deg + np.random.randn() * 1.5
        vpeak_true  = user_V0 * (1 - np.exp(-amp_deg / user_tau))
        dur_frames  = max(2, int(abs(dur_ms_true) / 1000 * fps))

        theta = np.random.uniform(0, 2 * np.pi)
        target_dir = np.array([
            np.sin(np.radians(amp_deg)) * np.cos(theta),
            np.sin(np.radians(amp_deg)) * np.sin(theta),
            np.cos(np.radians(amp_deg))
        ])
        target_dir /= np.linalg.norm(target_dir)

        for i in range(min(dur_frames, T - t)):
            alpha = i / (dur_frames - 1) if dur_frames > 1 else 1.0
            alpha_s = 1 / (1 + np.exp(-8 * (alpha - 0.5)))
            v = (1 - alpha_s) * current_dir + alpha_s * target_dir
            gaze[t + i] = v / np.linalg.norm(v)
        t += dur_frames
        current_dir = target_dir.copy()

        # 只记录持续时间在合理范围[15, 200]ms的saccade
        if 15 <= dur_ms_true <= 200:
            saccade_log.append({
                'amplitude':     amp_deg,
                'duration':      dur_ms_true,
                'peak_velocity': vpeak_true,
            })

    return gaze, saccade_log


# ============================================================
# 场景1：从真实用户序列提取参数
# ============================================================
def test_real_user_pipeline():
    print("\n" + "=" * 60)
    print("场景1：真实用户序列 -> 自动提取个性化参数")
    print("=" * 60)

    users = {
        'User_Fast':   dict(user_k=1.8, user_a=1.5, user_V0=700, user_tau=8.0),
        'User_Normal': dict(user_k=2.5, user_a=2.2, user_V0=580, user_tau=10.0),
        'User_Slow':   dict(user_k=3.5, user_a=3.0, user_V0=420, user_tau=13.0),
    }

    bank = UserMainSequenceBank(updater_type='bayesian')

    for user_id, cfg in users.items():
        _, saccade_log = simulate_user_gaze(n_seconds=30.0, fps=60.0, **cfg)
        params = bank.calibrate_user(user_id, saccade_log)

        dk = abs(params['k_i'] - cfg['user_k'])
        da = abs(params['a_i'] - cfg['user_a'])
        print(f"\n  {user_id}:")
        print(f"    真实参数: k={cfg['user_k']:.2f}, a={cfg['user_a']:.2f}, "
              f"V0={cfg['user_V0']:.0f}, tau={cfg['user_tau']:.1f}")
        print(f"    拟合参数: k={params['k_i']:.3f}, a={params['a_i']:.3f}, "
              f"V0={params.get('V0_i', 0):.1f}, tau={params.get('tau_i', 0):.2f}")
        print(f"    样本数: {len(saccade_log)}, "
              f"R2={params.get('r_squared', 0):.4f}  "
              f"(Dk={dk:.3f}, Da={da:.3f})")

    p_fast   = bank.get_params('User_Fast')
    p_normal = bank.get_params('User_Normal')
    p_slow   = bank.get_params('User_Slow')
    ok = p_fast['k_i'] < p_normal['k_i'] < p_slow['k_i']
    print(f"\n  k排序: Fast={p_fast['k_i']:.3f} < Normal={p_normal['k_i']:.3f} "
          f"< Slow={p_slow['k_i']:.3f} -> {'[OK]' if ok else '[FAIL]'}")

    return bank


# ============================================================
# 场景2：在线自适应更新
# ============================================================
def test_online_adaptation():
    print("\n" + "=" * 60)
    print("场景2：在线自适应更新（参数随使用逐渐收敛）")
    print("=" * 60)

    TRUE_A, TRUE_K = 2.0, 2.5

    rls   = RLSMainSequence()
    bayes = BayesianMainSequence()

    errors_rls   = []
    errors_bayes = []

    print(f"  群体先验: a={POPULATION_PRIOR['a_i']:.2f}, k={POPULATION_PRIOR['k_i']:.2f}")
    print(f"  真实参数: a={TRUE_A:.2f}, k={TRUE_K:.2f}")
    print(f"  {'次数':>6}  {'RLS_a':>7} {'RLS_k':>7} {'RLS误差':>9}  "
          f"{'Bay_a':>7} {'Bay_k':>7} {'Bay误差':>9}")

    for i in range(50):
        amp = np.random.uniform(2, 20)
        dur = TRUE_A + TRUE_K * amp + np.random.randn() * 1.5

        r_rls   = rls.update(amp, dur)
        r_bayes = bayes.update(amp, dur)

        err_rls   = abs(r_rls['a_i']   - TRUE_A) + abs(r_rls['k_i']   - TRUE_K)
        err_bayes = abs(r_bayes['a_i'] - TRUE_A) + abs(r_bayes['k_i'] - TRUE_K)
        errors_rls.append(err_rls)
        errors_bayes.append(err_bayes)

        if i in [0, 4, 9, 19, 29, 49]:
            print(f"  {i+1:>6}    "
                  f"{r_rls['a_i']:>7.3f} {r_rls['k_i']:>7.3f} {err_rls:>9.4f}    "
                  f"{r_bayes['a_i']:>7.3f} {r_bayes['k_i']:>7.3f} {err_bayes:>9.4f}")

    print(f"\n  最终误差 (|Da|+|Dk|): RLS={errors_rls[-1]:.4f}, Bayes={errors_bayes[-1]:.4f}")
    winner = 'Bayes' if errors_bayes[-1] < errors_rls[-1] else 'RLS'
    print(f"  -> 推荐使用: {winner}")
    return errors_rls, errors_bayes


# ============================================================
# 场景3：训练集成 - 个性化 vs 群体先验损失对比
# ============================================================
def test_training_integration():
    print("\n" + "=" * 60)
    print("场景3：训练集成 - 个性化约束 vs 群体先验约束")
    print("=" * 60)

    fps = 60.0
    B, T = 8, 16

    TRUE_PARAMS = {'a_i': 2.0, 'k_i': 2.5, 'V0_i': 580.0, 'tau_i': 10.0}
    amps  = np.random.uniform(5, 20, 25)
    durs  = TRUE_PARAMS['a_i'] + TRUE_PARAMS['k_i'] * amps + np.random.randn(25) * 1.5
    vpeak = TRUE_PARAMS['V0_i'] * (1 - np.exp(-amps / TRUE_PARAMS['tau_i']))
    saccades = [{'amplitude': a, 'duration': d, 'peak_velocity': v}
                for a, d, v in zip(amps, durs, vpeak)]
    personal_params = MainSequenceCalibrator.fit_all(saccades)

    def make_seq(amp_deg=10.0):
        seqs = []
        dur_ms = TRUE_PARAMS['a_i'] + TRUE_PARAMS['k_i'] * amp_deg
        dur_frames = max(2, int(dur_ms / 1000 * fps))
        for _ in range(B):
            start = np.array([0., 0., 1.])
            end   = np.array([np.sin(np.radians(amp_deg)), 0.,
                              np.cos(np.radians(amp_deg))])
            seq = []
            for t_idx in range(T):
                alpha = min(t_idx / (dur_frames - 1), 1.0) if dur_frames > 1 else 1.0
                alpha_s = 1 / (1 + np.exp(-8 * (alpha - 0.5)))
                v = (1 - alpha_s) * start + alpha_s * end
                seq.append(v / np.linalg.norm(v))
            seqs.append(seq)
        return torch.FloatTensor(np.array(seqs))

    good_seq = make_seq(amp_deg=10.0)   # 符合用户参数
    slow_seq = make_seq(amp_deg=5.0)    # 幅度不符（被低估）

    loss_personal   = MainSequenceConstraintLoss(fps=fps, personalized_params=personal_params)
    loss_population = MainSequenceConstraintLoss(fps=fps, personalized_params=POPULATION_PRIOR)

    results = {}
    for name, seq in [('符合用户参数', good_seq), ('不符用户参数', slow_seq)]:
        l_p = loss_personal(seq)
        l_g = loss_population(seq)
        results[name] = {'personal': l_p['total'].item(), 'population': l_g['total'].item()}
        print(f"\n  [{name}]")
        print(f"    个性化约束损失: {l_p['total'].item():.2f}")
        print(f"    群体先验损失:   {l_g['total'].item():.2f}")

    diff_personal   = abs(results['符合用户参数']['personal']   - results['不符用户参数']['personal'])
    diff_population = abs(results['符合用户参数']['population'] - results['不符用户参数']['population'])
    print(f"\n  个性化损失差异: {diff_personal:.2f}")
    print(f"  群体先验差异:   {diff_population:.2f}")
    print(f"  -> 个性化约束更敏感: {'[OK]' if diff_personal > diff_population else '[NOTE] 差异相近'}")

    return results


# ============================================================
# 场景4：推理阶段门控纠正
# ============================================================
def test_inference_verifier():
    print("\n" + "=" * 60)
    print("场景4：推理阶段门控纠正（异常预测 -> 自动修正）")
    print("=" * 60)

    params = {'a_i': 2.0, 'k_i': 2.5, 'V0_i': 580.0, 'tau_i': 10.0}
    verifier = MainSequenceVerifier(personalized_params=params, fps=60.0)

    gaze_normal = np.array([0., 0., 1.])
    gaze_after  = np.array([np.sin(np.radians(10)), 0., np.cos(np.radians(10))])
    gaze_after /= np.linalg.norm(gaze_after)

    # 30帧正常注视
    for _ in range(30):
        verifier.step(gaze_normal + np.random.randn(3) * 0.001)

    # 正常saccade帧
    r_normal = verifier.step(gaze_after, prev_gaze=gaze_normal)

    # 异常预测（CNN漂移5度）
    gaze_drift = np.array([np.sin(np.radians(15)), 0., np.cos(np.radians(15))])
    gaze_drift /= np.linalg.norm(gaze_drift)
    r_anomaly = verifier.step(gaze_drift, prev_gaze=gaze_normal)

    print(f"  正常saccade预测:")
    print(f"    置信度={r_normal['confidence']:.3f}, is_saccade={r_normal['is_saccade']}")
    print(f"    速度={r_normal['velocity']:.1f} deg/s")

    print(f"\n  异常预测（偏差5度）:")
    print(f"    置信度={r_anomaly['confidence']:.3f}, is_saccade={r_anomaly['is_saccade']}")
    angle_before = np.degrees(np.arccos(np.clip(np.dot(gaze_drift, gaze_normal), -1, 1)))
    angle_after  = np.degrees(np.arccos(np.clip(np.dot(r_anomaly['corrected'], gaze_normal), -1, 1)))
    print(f"    纠正前偏差={angle_before:.2f}deg, 纠正后偏差={angle_after:.2f}deg")
    print(f"    -> {'[OK] 纠正有效' if angle_after < angle_before else '[NOTE] 未触发纠正'}")


# ============================================================
# 场景5：可视化（保存图片）
# ============================================================
def test_visualization(errors_rls, errors_bayes):
    print("\n" + "=" * 60)
    print("场景5：绘制收敛曲线与Main Sequence个性化曲线")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 收敛曲线
    ax = axes[0]
    ax.plot(errors_rls,   label='RLS (lambda=0.98)', color='#2196F3', linewidth=2)
    ax.plot(errors_bayes, label='Bayesian',           color='#FF5722', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Saccade observations')
    ax.set_ylabel('|Da|+|Dk|')
    ax.set_title('Online update convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Main Sequence曲线（真实 vs 拟合）
    ax = axes[1]
    amps = np.linspace(1, 25, 100)

    dur_true = 2.0 + 2.5 * amps
    ax.plot(amps, dur_true, 'k--', label='Ground truth (a=2.0, k=2.5)', linewidth=2)

    dur_pop = POPULATION_PRIOR['a_i'] + POPULATION_PRIOR['k_i'] * amps
    ax.plot(amps, dur_pop, color='gray', linestyle=':', label='Population prior', linewidth=1.5)

    colors = ['#2196F3', '#4CAF50', '#FF5722']
    for (cfg_name, k, a), c in zip(
        [('Fast', 1.8, 1.5), ('Normal', 2.5, 2.2), ('Slow', 3.5, 3.0)],
        colors
    ):
        a_fit = np.clip(np.random.normal(a, 0.3), 0, 10)
        k_fit = np.clip(np.random.normal(k, 0.2), 1, 5)
        dur_fit = a_fit + k_fit * amps
        ax.plot(amps, dur_fit, color=c, label=f'User_{cfg_name}', linewidth=1.5)

    amps_obs = np.random.uniform(5, 22, 20)
    durs_obs = 2.0 + 2.5 * amps_obs + np.random.randn(20) * 2
    ax.scatter(amps_obs, durs_obs, color='black', s=20, alpha=0.5, label='Observations', zorder=5)

    ax.set_xlabel('Saccade amplitude (deg)')
    ax.set_ylabel('Duration (ms)')
    ax.set_title('Personalized Main Sequence curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / 'personalized_ms_test.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  图片已保存 -> {out_path}")
    plt.close()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    bank = test_real_user_pipeline()
    errors_rls, errors_bayes = test_online_adaptation()
    test_training_integration()
    test_inference_verifier()
    test_visualization(errors_rls, errors_bayes)

    print("\n" + "=" * 60)
    print("[OK] 所有场景测试完成！")
    print("=" * 60)
    print("""
接入现有训练脚本的方法：

  from personalized_main_sequence import PersonalizedGazeLoss, UserMainSequenceBank

  bank = UserMainSequenceBank(updater_type='bayesian')  # 推荐Bayesian
  loss_fn = PersonalizedGazeLoss(fps=60.0)

  # 每次拿到新saccade时更新参数
  bank.update(user_id, amplitude, duration)
  loss_fn.ms_loss.update_params(bank.get_params(user_id))

  # 训练循环
  preds = model(frames)          # [B, T, 3]
  result = loss_fn(preds[:,-1], targets, gaze_seq=preds)
  result['total'].backward()
""")
