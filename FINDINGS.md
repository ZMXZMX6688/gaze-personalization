# 迭代与优化findings（2026-07-07）

在服务器 `luxliang@192.168.1.85`（8× RTX 4090, `pytorch_env`: torch 2.4.1+cu121）
上对本仓库做了一轮系统的迭代、修复与实验验证。以下为结论与证据。

---

## 一、修复的正确性缺陷（已验证）

### 1. `SaccadeDetector` 末尾眼跳被静默丢弃
`detect()` 的状态机在序列结束时若仍处于 saccade 中，不会 flush 最后一个事件。
新增 `_finalize()` 辅助 + 循环后 flush。回归测试 `test_detector_flushes_trailing_saccade`。

原自测「注入3个saccade只检测到2个」实为**测试数据生成 bug**（累积位置，非
中心外跳），检测器本身在干净数据上可精确恢复幅度。已重写 Test6 为中心外跳+
缓慢回中，现 3/3 精确检测（A=10/15/8）。

### 2. `RLSMainSequence` 在线更新截距发散
- 现象：真值 a=2.0，原实现更新 30 次后 a 漂到 ~1.2–1.5 且随更新**变差**（Δa 0.14→1.23），
  最终比群体先验还差。
- 根因：λ=0.98 遗忘因子对**弱可辨识的截距**（幅度∈[2,20]°，从不接近 0°）放大方差，
  且 S=xᵀPx+1.0 与 P/λ 混用两套约定。
- 修复：改为一致的指数加权 RLS（S=λ·R+xᵀPx；P←(P−KxᵀP)/λ），默认 **λ=1.0**
  （平稳用户收敛到 OLS）、`obs_noise=5.0`、`P_init=1.0` 向群体先验正则化收缩，
  加协方差迹 anti-windup。
- 结果（200 次仿真平均 |Δa|+|Δk|）：**0.153，与 Bayesian 参照完全一致**（原 ~0.41）。
- 回归测试 `test_rls_converges_and_beats_prior`（阈值 0.30）、`test_rls_matches_bayesian`。

### 3. `SaccadeDetector` 缺少生理峰速上限
真实 60fps 数据存在 8000°/s 的单帧跳变（眨眼/跟踪丢失），会被当成 saccade 污染标定。
新增 `max_peak_velocity=1000°/s`（真实 saccade 峰速 ≈500–700°/s）拒绝伪影。

### 4. pytest 套件
`tests/test_core.py` 15 个用例覆盖检测器/标定/在线更新/损失/验证器/用户库，
含上述回归守卫。**15 passed**。（服务器 pytorch_env 已装 pytest 8.3.5 via aliyun 镜像）

---

## 二、实验验证：个性化 Main Sequence 约束在本数据上**无效**（关键负结果）

数据：`~/datasets/EXPORT_PUPIL_ALL`，56 段 TEyeD 格式 60fps AR 注视序列。
checkpoint：`resnet18_gru_bio_final_best.pt`（架构与 `ResNet18GRUBio` 精确匹配，
strict load missing=0/unexpected=0；识别训练配置 240/8/32，median 角误差 2.3°）。

### 2.1 训练时 MS 约束是双重 no-op
1. **无梯度**：`ablation_teyed_with_ms.py` 中 `pseudo_seq` 在 `torch.no_grad()` 下计算，
   `ms` 项对模型参数梯度恒为 0，实际只有 bio 角度约束在训练。
2. **采样几何**：clip 4 帧间隔 800ms（跨 2.4s），saccade 仅 20–80ms（≤单间隙 10%），
   完全不可见；MS 损失算的是 0.8s 尺度粗位移，非 saccade 动态。

### 2.2 离线标定无信号
调优检测器后（v_thresh=50, min_dur=10ms, max_peak_vel=900）每视频检出 200–640 个
saccade，但 **duration–amplitude 拟合 R²=0.0–0.13**，a/V0 顶到参数上界。
根因：60fps 下 saccade 时长（1.2–4.8 帧）量化过粗，且 AR 数据以微跳为主，主序列关系被抹平。
与作者原注释「低帧率采样下saccade时长量化误差大，自动检测不可靠」一致。

### 2.3 推理时 MS 验证：Δmean ≈ −0.02°（略微有害）
用 checkpoint 在密集 60fps 序列上逐帧滑窗预测（内部 clip=240/8/32），
`MainSequenceVerifier` 用 per-user / 群体先验参数纠正：

| 帧集合 | baseline mean | verify(user) | verify(pop) |
|---|---|---|---|
| 全部 8000 帧 | 4.587° | 4.607° | 4.615° |
| GT-saccade 帧(AR_36) | 37.5° | 37.8° | 37.9° |

- 聚合 Δmean(user)=**−0.020°**、Δmean(pop)=**−0.028°**（负=变差）。
- saccade 帧误差尾部（mean 37.5°, max 129.6°）由 **GT 眨眼/跟踪丢失伪影**主导
  （129° 注视跳变物理上不可能），MS 约束不能也不应"纠正"。
- 验证器偶尔**放大**最大误差（8°→27°）：在预测噪声上误触发并向错标定轨迹 slerp。

### 结论
个性化 Main Sequence 约束在本 60fps AR 数据上（训练时/离线标定/推理验证三条路径）
**均无正收益**。机理层面已完全刻画：模型在注视段已足够准；误差尾部是数据伪影；
60fps 无法分辨 saccade 动态以支撑标定。**要真正验证该方法需 ≥250fps 高速眼动数据。**
本仓库的检测器/标定/RLS-Bayesian/验证器/损失现已正确且有测试覆盖，可用于此类高帧率数据。

---

## 二·补：误差尾部 = GT 眨眼/跟踪丢失伪影（这才是能降 mean 的方向）

对密集逐帧预测按 `*validity_pupil.txt` 拆分误差（`eval_tail.py`）：

| 视频 | raw mean | **valid-only mean** | invalid-only mean | invalid 占比 | top-5%误差中invalid占比 |
|---|---|---|---|---|---|
| AR_36 | 7.14° | **2.41°** | 126.1° | 3.8% | **76.5%** |
| AR_39 | 2.03° | 2.03° | — | 0.0% | 0% |
| 聚合 | 4.59° | **2.22°** | — | 1.9% | — |

- **模型在有效帧上的真实误差 ≈ 2.2° mean**，raw ~4.6° 被 GT 伪影**放大约 2×**。
  任何密集逐帧 benchmark 不过滤 `validity==1` 都会高估误差约一倍。
  （仓库 ablation 建 clip 时已要求整窗 validity==1，故其报告数已是干净的；差异只出现在无过滤的密集评测。）
- **这正解释了 MS 验证为何无效**：它瞄准的误差尾部 76% 是坏标签，非任何约束能修的模型误差。
- 运行时（无 GT）用 prediction-jump 作弃权信号只召回 13–17% 的无效帧（GRU 平滑掉了眨眼），
  故正确做法是用 validity 标志过滤评测，而非基于预测的置信度。

## 三、可复现脚本（服务器 `~/gaze-personalization/`）
- `diag.py`：RLS 配置扫描 + 检测器合成验证
- `det_diag.py`：真实数据速度分布 / saccade 计数 vs 阈值
- `calib_check.py`：调优检测器的真实数据标定（看是否顶界）
- `config_id.py`：由角误差识别 checkpoint 训练配置
- `eval_verify.py`：密集推理 + MS 验证对比（`GAZE_SIDS/WLEN/NVID/W0` 可配）
- `eval_tail.py`：误差尾部按 validity 拆分（valid-only vs 伪影）+ 运行时弃权信号评估
