# Gaze Personalization：3D 视线估计个性化

本项目研究一个实际问题：同一个 3D 视线估计模型面对不同用户时，眼球结构、注视习惯和拍摄条件会带来个人误差。项目先训练一个适用于全部用户的通用模型，再用新用户少量带标签的校准片段拟合轻量适配器，从而降低该用户的视线角误差。

当前实验基于 **TEyeD 完整导出数据**，共包含 56 个用户。主模型为 ResNet18-GRU，输入眼部视频片段，输出三维视线方向。

新同学参与项目时，可先阅读 [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)，其中包含实时实验状态、服务器目录、待认领任务和协作规范。

## 项目如何工作

完整流程可以概括为四步：

1. 用训练用户的数据训练通用 3D 视线模型。
2. 冻结通用模型的全部参数。
3. 为新用户采集 K 个带标签校准片段，并拟合一个小型输出适配器。
4. 在独立评测片段上同时计算校准前、校准后的角度误差。

当前实现了三种适配器：

| 适配器 | 参数量 | 作用 |
|---|---:|---|
| yaw/pitch bias | 2 | 修正水平和垂直方向的固定偏差 |
| SO(3) rotation | 3 | 对三维视线方向做小角度旋转 |
| tangent affine | 6 | 修正偏移、缩放和水平/垂直方向之间的耦合 |

项目中的常用术语：

| 术语 | 含义 |
|---|---|
| K | 每位新用户参与拟合的校准片段数量 |
| Baseline | 通用模型在个性化校准前的误差 |
| Personalized | 加载用户适配器后的误差 |
| Gain | Baseline 减去 Personalized，数值越大表示改善越明显 |
| Macro mean | 先计算每位用户的平均误差，再对用户取平均 |
| Chronological | 使用记录前段做校准，使用后段做评测 |
| Interleaved | 在整段记录中分散选择校准片段，并隔离相邻评测片段 |

## 当前实验结果

一句话总结：**56 用户五折实验已完成；chronological affine 在 K=50 时将宏平均误差从 1.1754° 降至 1.1226°，改善 0.0528°（4.49%）。**

| 实验范围 | 个性化方案 | Baseline | Personalized | Gain | 相对改善 |
|---|---|---:|---:|---:|---:|
| **56 用户五折** | **chronological affine，K=50** | **1.1754°** | **1.1226°** | **0.0528°** | **4.49%** |
| 56 用户五折 | chronological affine，K=20 | 1.1754° | 1.1289° | 0.0465° | 3.95% |
| 56 用户五折 | chronological bias，K=20 | 1.1754° | 1.1323° | 0.0431° | 3.67% |
| 56 用户五折 | interleaved bias，K=10 | 1.1929° | 1.1528° | 0.0402° | 3.37% |
| 早期固定 6 人 | chronological affine，K=40 | 1.2176° | 1.1504° | 0.0672° | 5.52% |
| 早期固定 6 人、无约束版 | interleaved bias，K=50 | 0.9180° | 0.8992° | 0.0188° | 2.05% |

结果可以这样理解：

- **总体结论：** 56 位用户均作为测试用户参与一次，五折汇总覆盖全部用户；K=50 的 chronological affine 获得最大宏平均改善。
- **样本效率：** K=20 的 chronological affine 在 10 次重复中平均改善 0.0465°，Gain 的 5% / 95% 分位数为 +0.0381° / +0.0558°。
- **轻量方案：** 仅 2 个参数的 chronological bias 在 K=20 时也改善 0.0431°，与 affine 的差距较小。
- **小样本风险：** affine 在 K=5 时平均退化 0.0463°；小校准集优先使用 bias 或让门控回退到通用预测。
- **历史最低值：** 0.8992° 来自固定 6 人的无约束 checkpoint 实验，不应与覆盖 56 人、每折独立训练 checkpoint 的五折结果直接横向比较。

五折实验使用 C1/C2/C3 约束训练路径、subject-disjoint 划分、3 种适配器、2 种协议、K∈{5,10,20,50}，每个配置重复 10 次，共汇总 13,020 条结果。K=50 因样本不足时覆盖率为 92.86%–96.43%；K=20 的覆盖率为 96.43%。

下面保留固定 6 位留出用户的早期实验细节，用于追踪方法演进；当前总体结论以 56 用户五折结果为准。

<details>
<summary>查看 chronological affine 的完整 K 曲线</summary>

这组实验使用 C1/C2/C3 约束版 checkpoint。每位用户预留 50 个早期校准片段，评测使用后续片段。每个 K 进行 50 次分层重复抽样，K=40 另外完成了 100 次确认实验。

| K | Personalized | 平均 Gain | 重复标准差 | Gain 的 5% 分位数 |
|---:|---:|---:|---:|---:|
| 5 | 1.2370° | -0.0194° | 0.0484° | -0.1172° |
| 10 | 1.1851° | 0.0326° | 0.0334° | -0.0246° |
| 15 | 1.1713° | 0.0463° | 0.0243° | 0.0042° |
| 20 | 1.1582° | 0.0594° | 0.0126° | 0.0425° |
| 25 | 1.1555° | 0.0621° | 0.0149° | 0.0423° |
| 30 | 1.1607° | 0.0569° | 0.0133° | 0.0391° |
| 40 | **1.1508°** | **0.0668°** | 0.0032° | 0.0620° |
| 50 | 1.1583° | 0.0593° | 0.0000° | 0.0593° |

K=40 的 100 次确认结果：

- Baseline：1.2176°
- Personalized：1.1504°
- 平均 Gain：0.0672°
- 重复标准差：0.0033°
- Gain 的 5% / 95% 分位数：+0.0619° / +0.0725°
- 有效改善比例：80.0%，有效改善阈值为 0.001°

曲线显示 K≥15 后，Gain 的 5% 分位数保持为正；K=40 当前兼顾最终误差与重复稳定性。

</details>

<details>
<summary>查看早期单次 K=20 结果</summary>

早期 deterministic interleaved bias 实验在 K=20 时得到 1.2076° → 1.1543°，Gain 为 0.0533°（4.4%）。加入 20 次分层重复抽样后，平均 Gain 为 0.0320°。该记录用于展示校准样本选择带来的波动，当前主结论采用大规模重复实验。

</details>

## 56 用户五折实验

五折 subject-disjoint 交叉验证已于 2026-07-19 完成，56 个用户各自作为测试用户一次：

- 5 个 fold 均完成通用模型训练与个性化评测；
- 汇总包含 56 位用户和 13,020 条逐用户、逐重复结果；
- 评测覆盖 bias、rotation、affine，以及 chronological、interleaved 两种协议；
- 服务器汇总目录为 `/data1/luxliang/gaze-personalization/cv5/aggregate`。

当前五折任务评估 C1/C2/C3 约束训练路径。无约束通用模型仍只有固定 6 人对照，后续需要补充同口径五折实验。

## 快速运行

### 1. 运行单个 checkpoint 的个性化评测

`personalize_from_universal.py` 会复现 subject-disjoint 数据划分，冻结通用模型，为每位测试用户拟合 yaw/pitch bias，并保存完整的校准与评测来源。

```bash
python3 personalize_from_universal.py \
  --data-dir /path/to/EXPORT_PUPIL_ALL \
  --checkpoint /path/to/resnet18_gru_bio_two_stage_best.pt \
  --device cuda \
  --calibration-sizes 5,10,20,50 \
  --split-strategy interleaved
```

默认 TEyeD 索引参数为：

- `segment_min_len=120`
- 每位用户最多 60 个 segment
- 每个 segment 均匀提取 15 个 clip
- 数据划分随机种子为 `seed=42`

在当前 56 用户导出数据上，该配置得到 4,500 个验证 clips 和 5,400 个测试 clips。

### 2. 比较多种适配器

`personalization_benchmark.py` 会先缓存通用模型预测，再比较 bias、rotation 和 affine。缓存可以减少视频重复解码，适合大规模 K 扫描和重复抽样。

```bash
python3 personalization_benchmark.py \
  --data-dir /path/to/EXPORT_PUPIL_ALL \
  --checkpoint /path/to/resnet18_gru_bio_two_stage_best.pt \
  --cache-dir /large_disk/gaze-personalization-cache \
  --output-dir /large_disk/gaze-personalization-benchmark \
  --device cuda:0 \
  --methods bias,rotation,affine \
  --protocols chronological,interleaved \
  --calibration-sizes 5,10,20,50 \
  --repeats 20
```

输出文件包括：

- `results.csv`：每位用户、每次重复的结果
- `summary.csv`：各实验配置的汇总指标
- `subject_summary.csv`：逐用户稳定性统计
- `summary.json`：实验参数与总体结果
- `split_manifest.json`：校准和评测片段的来源记录

### 3. 运行 subject-disjoint 交叉验证

`make_subject_cv_folds.py` 用于生成确定性的用户级交叉验证划分。`train_ours_two_stage.py --split-json ... --fold-index ...` 根据指定 fold 训练通用模型，并将用户划分写入 checkpoint 目录。

## 数据隔离

个性化实验采用 segment 级数据隔离：

- 校准集与评测集来自不同 segment；
- interleaved 协议会排除校准 segment 的相邻 segment；
- `split_manifest.json` 记录每个 clip 的 segment 和源帧；
- calibration-validation gate 根据校准内部验证结果选择适配强度；
- 证据较弱时，gate 选择零强度，输出保持通用模型预测。

通用约束版 checkpoint 的 SHA-256：

```text
be2c9951c02543f262d3413c9e170b303592e47922e607af5444893ca8376564
```

## 代码结构

- `personalize_from_universal.py`：冻结通用模型并完成新用户校准
- `personalization_benchmark.py`：多适配器、大规模重复实验
- `make_subject_cv_folds.py`：生成 subject-disjoint 五折划分
- `train_ours_two_stage.py`：训练通用模型与实验性用户条件分支
- `personalized_main_sequence.py`：个性化 Main Sequence 研究代码
- `ablation_teyed_with_ms.py`：TEyeD Main Sequence 历史消融
- `tests/`：单元测试与回归测试
- `FINDINGS.md`：详细实验记录、逐用户结果与问题分析

## Main Sequence 分支

Main Sequence 描述眼跳幅度与峰值速度之间的关系。可靠拟合通常需要能够解析 20-80 ms 眼跳过程的高帧率数据。当前 TEyeD 导出为 60 fps，离线拟合的 R² 接近 0，推理验证也没有带来角度误差改善。

当前正式 TEyeD 个性化路径采用冻结通用模型加轻量输出适配器。Main Sequence 分支保留为高帧率数据研究方向，建议采集帧率达到 250 fps 或更高。

## 测试

```bash
python3 -m pytest -q
```

当前测试集包含 30 个测试，覆盖适配器恒等初始化、SO(3)/affine 恢复、门控回退、用户级交叉验证、GRU 隐状态布局，以及两种协议的 segment/frame 隔离。

## 服务器与认证

`upload_and_run.py` 通过本机 SSH agent/key 调用 `scp` 和 `ssh`。服务器地址、用户、数据目录和 checkpoint 可以通过命令行参数或以下环境变量配置：

- `GAZE_HOST`
- `GAZE_USER`
- `EYE_DATA_DIR`
- `UNIVERSAL_CHECKPOINT`
