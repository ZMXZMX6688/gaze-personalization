# 迭代与优化findings（2026-07-07）

## 2026-07-18：输出适配器大规模重采样审计

新增 `personalization_benchmark.py`，把 6 个未见用户的 5,400 个普适模型预测缓存一次，
随后在不重复解码视频的前提下比较三类小参数适配器：2 参数 yaw/pitch bias、3 参数
SO(3) rotation、6 参数 tangent affine。正式扫描覆盖 chronological/interleaved、
K∈{5,10,20,50}、20 次分层随机重采样，共 2,880 个“用户×方法×协议×K×重复”结果。

关键结果如下：

| protocol | adapter | K | base macro | personalized macro | improvement | repeat std | p05 | user-repeat win rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chronological | affine | 20 | 1.2176° | **1.1573°** | **0.0604°** | 0.0122° | +0.0448° | 74.2% |
| chronological | affine | 50 | 1.2176° | 1.1583° | 0.0593° | 0 | +0.0593° | 66.7% |
| interleaved | affine | 50 | 1.2076° | 1.1562° | 0.0514° | 0 | +0.0514° | 16.7% |
| interleaved | bias | 50 | 1.2076° | 1.1618° | 0.0458° | 0 | +0.0458° | 50.0% |
| interleaved | bias | 20 | 1.2076° | 1.1756° | 0.0320° | 0.0154° | ≈0° | 25.0% |

原单次 interleaved K=20 bias 的 0.0533° 提升在随机重采样后降为 0.0320°，说明旧结果
对 calibration clip 的选择有明显敏感性。interleaved affine K=50 虽然宏平均较好，但主要由
单个用户贡献，不能作为稳定普适结论。

新的稳定候选是 **chronological affine K=20**：固定 50-clip early calibration pool 后，
进一步做 100 次重采样，平均改善 **0.0595°**，标准差 0.0142°，5% 分位仍为
**+0.0391°**。以至少 0.001° 为有效改善阈值，73.3% 的全部用户-重复组合获益；在门控
实际启用适配器的组合中，87.5% 获益。该结果说明偏差不只是常数 yaw/pitch offset，
还包含个体化增益和轴间耦合；6 参数近恒等仿射比 2 参数 bias 更适合从早期校准外推到后续片段。

以上仍基于同一个普适 checkpoint 的 6 个 held-out SID。已经启动 5 折 subject-disjoint
训练与个性化评测，使 56 个 SID 各作为测试用户一次；在该交叉验证完成前，不把上述收益
升级为全数据集结论。

## 2026-07-15：普适模型到未见用户的个性化审计

- 原 `train_ours_two_stage.py` 只在训练批次的 `forward_all(subject_idx=...)`
  中使用 subject embedding，验证/测试调用的 `forward()` 会绕开 embedding，且没有为
  未见测试用户建立 calibration 参数。因此该流程不是未见用户个性化评测。
- `hidden_init` 原先直接把 `(B, 2H)` reshape 为 `(2, B, H)`，会在 batch size > 1
  时混合受试者和 GRU 层。现已改为先 reshape `(B, 2, H)` 再 permute。
- 新增 `personalize_from_universal.py`：严格加载并冻结普适 checkpoint，仅用与评测
  segment 不相交的标注 clip 拟合每用户 2 参数 yaw/pitch 偏置。
- 本地 56-SID 数据的原始 segment 中位长度为 356 帧。`segment_min_len=600` 会令
  验证/测试集为空；恢复为 `segment_min_len=120`、每段 15 个 clip 后得到
  36,363/4,500/5,400，验证和测试数量与历史日志完全一致。
- 收敛普适模型的 768-clip 探针中 C1/C2/C3 全部分支激活率为 0。初始化探针中仅
  C1 elevation 与 C2 激活，C3/flip/spherical/hinge 均为 0；现有约束消融也未显示收益。

正式个性化指标必须逐 SID 报告，并以未适配普适输出作为同一批 evaluation clip 上的
配对基线；不能把 calibration clip 混入 evaluation，也不能只报告 clip 加权总体均值。

### 未见用户个性化结果

冻结 checkpoint：`resnet18_gru_bio_two_stage_best.pt`，SHA-256
`be2c9951c02543f262d3413c9e170b303592e47922e607af5444893ca8376564`。测试集为普适模型
固定的 6 个 held-out SID，每人 50 个 calibration-pool clips；adapter 仅有 yaw/pitch
两个参数，并在 calibration 内部验证后选择 0/0.25/0.5/0.75/1.0 缩放，证据不足时回退
到零偏置。

严格 `chronological` pilot（每人 150 个 evaluation clips）在 K=50 时由 1.2076° 变为
1.2121°，宏平均下降 0.0045°。这说明记录开头的少量校准不能稳定外推到后续时段，不能
把该协议宣称为有效个性化。

`interleaved` session-wide 协议把 calibration segments 均匀分布到整段记录，并排除每个
calibration segment 及其前后一个相邻 segment。完整评测每人 750 个 clips，segment/frame
严格隔离，结果如下：

| K | universal macro mean | personalized macro mean | improvement | subject win rate |
|---:|---:|---:|---:|---:|
| 5 | 1.2076° | 1.1800° | 0.0276° | 2/6 |
| 10 | 1.2076° | 1.1916° | 0.0160° | 1/6 |
| 20 | 1.2076° | **1.1543°** | **0.0533°** | 3/6 |
| 50 | 1.2076° | 1.1618° | 0.0458° | 3/6 |

其余用户由 calibration-validation 门控回退到普适输出，没有实质负迁移。该正结果支持
“一次 session 内分散校准”，不等价于对未来时段的跨时间泛化。完整产物位于服务器
`~/gaze-personalization/runs/personalization-interleaved-full-20260715/`。

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
