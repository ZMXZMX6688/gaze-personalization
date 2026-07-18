# Gaze Personalization 项目进度与协作指南

> 更新时间：2026-07-18（Asia/Shanghai）  
> 项目仓库：`git@github.com:ZMXZMX6688/gaze-personalization.git`  
> 主分支：`main`  
> 本文用途：帮助新同学在较短时间内了解目标、结果、运行任务和可认领工作。

## 1. 项目目标

项目面向基于眼部视频的 3D 视线估计。通用模型在不同用户上会受到眼球结构、注视习惯和采集条件影响，因此项目加入少样本个性化校准：

1. 训练 ResNet18-GRU 通用视线模型；
2. 冻结通用模型；
3. 使用新用户 K 个带标签片段拟合轻量输出适配器；
4. 在独立片段上评估校准前后的角度误差。

当前数据集为 TEyeD 完整导出数据，共 56 个用户。正式个性化路线包含 2 参数 bias、3 参数 SO(3) rotation 和 6 参数 tangent affine。

项目介绍和运行示例见 [README.md](README.md)，实验推导和历史问题见 [FINDINGS.md](FINDINGS.md)。

## 2. 当前进度概览

| 模块 | 状态 | 当前结论或产物 |
|---|---|---|
| TEyeD 数据核查 | 已完成 | 56 个 SID；默认索引得到 4,500 个验证 clips 和 5,400 个测试 clips |
| 新用户个性化协议 | 已完成 | 冻结通用模型，segment/frame 隔离，逐用户配对评测 |
| bias / rotation / affine | 已完成 | 三种轻量输出适配器已纳入统一 benchmark |
| 重复抽样 benchmark | 已完成 | 支持 K 扫描、两种时间协议、重复统计和门控 |
| 固定 6 用户实验 | 已完成 | 最低绝对误差 0.8992°；稳定改善 0.0672°（5.52%） |
| 56 用户五折训练 | 运行中 | fold 0-3 正在训练，fold 4 已排队 |
| 五折个性化评测 | 已排队 | 每折训练结束后自动运行 |
| 五折结果汇总 | 已排队 | 五折评测结束后自动生成 aggregate |
| 无约束模型五折对照 | 待启动 | 使用相同 fold 划分建立更强通用基线 |
| Main Sequence | 研究分支 | 60 fps 数据上的拟合信号较弱，面向后续高帧率数据 |

## 3. 已确认的实验结果

当前固定 6 位留出用户的主要结果：

| 通用模型 | 个性化方案 | Baseline | Personalized | Gain | 相对改善 |
|---|---|---:|---:|---:|---:|
| C1/C2/C3 约束版 | chronological affine，K=40 | 1.2176° | 1.1504° | 0.0672° | 5.52% |
| 无约束版 | interleaved bias，K=50 | 0.9180° | **0.8992°** | 0.0188° | 2.05% |

实验解释：

- 当前最低最终误差为 **0.8992°**，来自无约束通用模型加 interleaved bias。
- 当前最稳定的时间外推改善为 **0.0672°**，来自约束版通用模型加 chronological affine。
- 个性化收益会随通用 checkpoint 改变。无约束版在 chronological affine K=20 时误差增加约 0.0695°。
- 每项结论都需要同时报告 Baseline、Personalized 和 Gain。
- 以上数值属于固定 6 用户实验，56 用户总体结果等待五折 aggregate 完成。

chronological affine 的 K 扫描显示：K≥15 后 Gain 的 5% 分位数保持为正，K=40 当前兼顾误差与重复稳定性。K=40 的 100 次确认实验得到：

- Baseline：1.2176°
- Personalized：1.1504°
- 平均 Gain：0.0672°
- 重复标准差：0.0033°
- Gain p05 / p95：+0.0619° / +0.0725°
- 有效改善比例：80.0%，阈值为 0.001°

## 4. 正在运行的五折实验

### 4.1 进度快照

以下状态采集于 2026-07-18。`val_ang` 用于观察单折训练收敛，最终结论以五折个性化 aggregate 为准。

| Fold | GPU | 当前状态 | 最近完成轮次 | 当前 best val_ang |
|---:|---:|---|---|---:|
| 0 | 0 | Stage 2 训练中 | 6 / 9 | 1.546° |
| 1 | 3 | Stage 2 训练中 | 5 / 9 | 1.063° |
| 2 | 5 | Stage 2 训练中 | 6 / 9 | 1.418° |
| 3 | 6 | Stage 2 训练中 | 6 / 9 | 0.934° |
| 4 | 0 | 等待 fold 0 完成 | 已排队 | — |

自动任务链：

```text
fold 训练完成
  → 读取该 fold 的 best checkpoint
  → 运行 bias / rotation / affine 个性化 benchmark
  → 五个 fold 全部完成
  → aggregate_cv_personalization.py 汇总
```

### 4.2 实时查看命令

```bash
ssh luxliang@192.168.1.85
tmux list-sessions
tail -f /data1/luxliang/gaze-personalization/cv5/fold0.log
nvidia-smi
```

相关 tmux 会话：

- 训练：`gaze_cv_fold0`、`gaze_cv_fold1`、`gaze_cv_fold2`、`gaze_cv_fold3`
- fold 4 队列：`gaze_cv_fold4_queue`
- 个性化评测：`gaze_cv_eval0` 至 `gaze_cv_eval4`
- 汇总：`gaze_cv_aggregate`

### 4.3 完成标志

五折主任务完成后应出现以下产物：

```text
/data1/luxliang/gaze-personalization/cv5/fold0-personalization/summary.json
/data1/luxliang/gaze-personalization/cv5/fold1-personalization/summary.json
/data1/luxliang/gaze-personalization/cv5/fold2-personalization/summary.json
/data1/luxliang/gaze-personalization/cv5/fold3-personalization/summary.json
/data1/luxliang/gaze-personalization/cv5/fold4-personalization/summary.json
/data1/luxliang/gaze-personalization/cv5/aggregate/
```

## 5. 服务器与实验目录

| 资源 | 路径 |
|---|---|
| 服务器 | `luxliang@192.168.1.85` |
| 代码仓库 | `/home/luxliang/gaze-personalization` |
| Python 环境 | `/opt/anaconda3/envs/pytorch_env` |
| TEyeD 数据 | `/data1/luxliang/datasets/EXPORT_PUPIL_ALL` |
| 五折 split | `/data1/luxliang/gaze-personalization/cv5-folds-seed42.json` |
| 五折训练与评测 | `/data1/luxliang/gaze-personalization/cv5` |
| 五折最终汇总 | `/data1/luxliang/gaze-personalization/cv5/aggregate` |
| 固定 6 用户 benchmark | `/data1/luxliang/gaze-personalization/runs/benchmark-heldout6-20260718-full` |
| 无约束 benchmark | `/data1/luxliang/gaze-personalization/runs/benchmark-no-constraint-heldout6` |
| affine K 曲线 | `/data1/luxliang/gaze-personalization/runs/affine-chronological-calcurve-r50-v2` |
| affine K=40 复核 | `/data1/luxliang/gaze-personalization/runs/affine-chronological-pool50-k40-r100` |

服务器访问沿用个人 SSH key。项目文档只记录连接方式和共享实验路径。

## 6. 新同学快速接入

### 6.1 获取代码

```bash
git clone git@github.com:ZMXZMX6688/gaze-personalization.git
cd gaze-personalization
git switch main
git pull --ff-only origin main
```

服务器主仓库正在为 fold 4、逐折个性化评测和 aggregate 提供代码。五折任务结束前，该目录保持 `main` 代码冻结。查看主仓库状态：

```bash
ssh luxliang@192.168.1.85
cd /home/luxliang/gaze-personalization
git status --short
git log -3 --oneline
```

服务器开发统一使用独立 worktree：

```bash
cd /home/luxliang/gaze-personalization
git fetch origin
mkdir -p /data1/luxliang/gaze-personalization-worktrees
git worktree add \
  /data1/luxliang/gaze-personalization-worktrees/<任务编号> \
  -b task/<任务编号>-<简短名称> origin/main
cd /data1/luxliang/gaze-personalization-worktrees/<任务编号>
```

### 6.2 运行测试

```bash
/opt/anaconda3/envs/pytorch_env/bin/python -m pytest -q
```

当前回归测试数量为 30，覆盖适配器、数据隔离、五折划分、门控和 GRU 隐状态布局。

### 6.3 选择任务并创建分支

```bash
git switch -c task/<任务编号>-<简短名称>
```

认领任务时，在本文第 7 节填写负责人和分支名。实验输出使用独立目录，目录名建议包含任务编号、协议、K、重复次数和日期。

示例：

```text
/data1/luxliang/gaze-personalization/runs/P1-02-affine-k40-r100-20260718
```

### 6.4 提交前检查

```bash
python3 -m pytest -q
git diff --check
git status --short
```

提交内容应包含：

- 使用的 Git commit；
- checkpoint 路径与 SHA-256；
- split JSON、fold index 和随机种子；
- 完整运行命令；
- Baseline、Personalized、Gain；
- 逐用户结果和汇总结果路径；
- 对 README、FINDINGS 或本文档的结果更新。

## 7. 可认领任务

| 编号 | 优先级 | 状态 | 任务 | 完成标准 | 负责人 / 分支 |
|---|---|---|---|---|---|
| P0-01 | P0 | 自动运行中 | 完成约束版 56 用户五折训练与个性化评测 | 5 个 summary 完整，aggregate 成功生成 | 自动任务链 |
| P0-02 | P0 | 待认领 | 审计五折数据完整性 | 56 个 SID 各测试一次；校准、验证、评测来源可追踪；输出审计表 | 待认领 |
| P0-03 | P0 | 待认领 | 启动无约束通用模型五折对照 | 复用同一 split；保存每折 checkpoint；生成个性化 aggregate | 待认领 |
| P1-01 | P1 | 待认领 | 五折结果可视化 | 输出 fold、用户、K、adapter、protocol 维度图表及数据表 | 待认领 |
| P1-02 | P1 | 待认领 | 逐用户收益分析 | 分析 baseline 强弱、适配器启用率、用户 Gain 和失败案例 | 待认领 |
| P1-03 | P1 | 待认领 | checkpoint 敏感性分析 | 在约束版与无约束版上使用同一协议和随机抽样，对比绝对误差与 Gain | 待认领 |
| P1-04 | P1 | 待认领 | affine 校准样本效率复核 | 五折范围扫描 K=10/15/20/25/30/40/50，报告 p05/p50/p95 | 待认领 |
| P1-05 | P1 | 待认领 | 汇总脚本与文档自动化 | aggregate 自动生成 Markdown 表格和图表索引 | 待认领 |
| P2-01 | P2 | 待认领 | 高帧率 Main Sequence 数据方案 | 给出采集帧率、眼跳检测、拟合指标和验证协议 | 待认领 |

建议优先顺序：P0-02 → P0-03 → P1-01/P1-02 → P1-03/P1-04。

## 8. 实验统一规范

### 8.1 数据划分

- 用户划分采用 subject-disjoint；
- 校准和评测使用不同 segment；
- interleaved 协议排除校准 segment 的相邻 segment；
- chronological 协议使用早期片段校准、后续片段评测；
- split 和 clip 来源写入 `split_manifest.json`。

### 8.2 指标

主要指标为三维视线方向的角度误差，单位为度。汇总以 macro-subject mean 为主，同时保存：

- 每位用户 Baseline；
- 每位用户 Personalized；
- 每位用户 Gain；
- 重复抽样标准差；
- Gain 的 p05 / p50 / p95；
- 有效改善比例，当前阈值为 Gain > 0.001°。

### 8.3 结果解释

- Baseline 反映通用模型质量；
- Personalized 反映最终可用精度；
- Gain 反映同一 checkpoint 上的个性化贡献；
- checkpoint、adapter、protocol 和 K 共同决定结果；
- 固定 6 用户结果标记为 preliminary；
- 56 用户五折 aggregate 用于总体结论。

### 8.4 运行安全

- 启动 GPU 任务前查看 `nvidia-smi` 和 `tmux list-sessions`；
- `/home/luxliang/gaze-personalization` 为当前五折自动任务链的代码源，任务结束前保持 `main` 代码冻结；
- 现有 `/data1/luxliang/gaze-personalization/cv5` 目录由自动任务链管理；
- 新开发统一使用 `/data1/luxliang/gaze-personalization-worktrees/<任务编号>`；
- 新实验使用新的 output/cache 目录；
- checkpoint、cache 和 split manifest 保留到结果审计完成；
- 密码、Token 和私钥保留在个人凭据系统中。

## 9. 核心代码入口

| 文件 | 用途 |
|---|---|
| `train_ours_two_stage.py` | 训练 ResNet18-GRU 通用模型，支持显式 fold split |
| `personalize_from_universal.py` | 冻结通用模型并拟合单用户 bias |
| `personalization_benchmark.py` | 缓存预测并比较 bias、rotation、affine |
| `make_subject_cv_folds.py` | 生成确定性的 subject-disjoint 五折划分 |
| `aggregate_cv_personalization.py` | 汇总五折个性化结果 |
| `personalized_main_sequence.py` | Main Sequence 研究分支 |
| `tests/` | 单元与回归测试 |
| `FINDINGS.md` | 实验日志、问题分析和历史结果 |

## 10. 进度更新模板

完成一个实验后，将以下内容追加到本文档或 `FINDINGS.md`：

```markdown
### YYYY-MM-DD：任务编号与实验名称

- 负责人：
- Git commit：
- 数据与 split：
- checkpoint：
- 完整命令：
- 输出目录：
- Baseline：
- Personalized：
- Gain：
- 重复统计：
- 主要观察：
- 后续动作：
```

README 保持项目介绍和稳定结论，本文维护当前进度与协作任务，FINDINGS 保存详细实验过程。
