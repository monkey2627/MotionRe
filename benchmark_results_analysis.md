# 多方法 IMU 人体姿态重建基准分析报告

## 1. 执行摘要

本报告分析 `benchmark_results/` 中已经完成的基准结果，覆盖 8 种方法：GlobalPose、IMUCoCo、MobilePoser、PIP、PNP、SliMeVR、TransPose 和 WheelPoser。结果包含 DIP-IMU 常规评测（2,799 帧，30 FPS）以及基于 AMASS 动作清单的 drift 评测（3,600 帧，30 FPS，最长 120 秒）。

主要结论如下：

1. **DIP-IMU 旋转精度**：TransPose 最低（19.58°），PIP 次之（20.04°）；MobilePoser 仅使用 5 个传感器，误差 20.62°，与两种 6 传感器方法接近。
2. **DIP-IMU 腰部相关关节**：PIP（6.69°）和 TransPose（6.10°）最好，MobilePoser 为 8.20°，明显优于 PNP（16.29°）和 WheelPoser（24.09°）。
3. **AMASS drift 旋转稳定性**：PIP 最好（13.27°），TransPose（19.35°）和 MobilePoser（20.07°）随后。MobilePoser 在 5 传感器条件下保持了与 6 传感器 TransPose 相近的全身旋转误差。
4. **平移漂移是 MobilePoser 的主要短板**：DIP 平移误差 2.266 m，drift 为 12.766 m，均显著高于 PIP（0.322 m/1.005 m）和 TransPose（0.568 m/1.421 m）。因此 MobilePoser 当前更适合作为低传感器姿态旋转基线，不能直接宣称具备可靠的全局位移跟踪能力。
5. **异常动作覆盖尚未形成公平的逐动作比较**：现有标准结果是跨动作平均值；视频数量表明多数方法生成了 618 个 drift 视频，WheelPoser 也显示 11 类动作，但报告中没有统一的按动作误差 JSON，因此不能据此断言某个方法在 lying、crawling 或 transitions 上最鲁棒。
6. **任务完成性必须纳入结论**：GlobalPose、PIP 和 PNP 的 drift benchmark_report 标记为失败；其目录中残留的 `standard_metrics.json` 不能与完整成功运行的方法等价解读。失败原因需在服务器日志中进一步定位。

## 2. 评测定义与数据完整性

### 2.1 指标

- `mean_rotation_deg/all`：24 个 SMPL 关节的平均旋转角误差，单位为度，越低越好。
- `lumbar`：关节索引 3、6、9 的平均误差，用于观察躯干/腰部姿态。
- `hips`、`knees`、`upper_arms`、`forearms`：对应关节组平均误差。
- `mean_translation_m`：初始根节点对齐后的逐帧根平移欧氏误差，单位为米，越低越好。缺失值表示该方法未提供可比较的平移结果。
- 所有标准结果均以 `standard_metrics.json` 和 `standard_metrics.npz` 为准；视频仅用于定性检查。

### 2.2 样本规模

| 套件 | 有效帧数 | 采样率 | 解释 |
|---|---:|---:|---|
| DIP | 2,799 | 30 FPS | 统一的 DIP-IMU 测试片段 |
| drift | 3,600 | 30 FPS | AMASS 动作清单抽样，约 120 秒上限 |

drift 与 DIP 不是同一批样本，因此不能把 drift 数值简单视为 DIP 数值的时间退化曲线。drift 结果更适合衡量长时序和动作覆盖下的稳定性。

## 3. DIP-IMU 结果

| 方法 | 传感器数 | 全身旋转 (°) | 腰部 (°) | 髋部 (°) | 膝部 (°) | 上臂 (°) | 前臂 (°) | 平移 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TransPose | 6 | **19.58** | **6.10** | 11.04 | 44.57 | **22.81** | **34.37** | 0.568 |
| PIP | 6 | 20.04 | 6.69 | **16.70** | 38.26 | 26.70 | 37.58 | **0.322** |
| MobilePoser | **5** | 20.62 | 8.20 | 18.37 | **13.46** | 33.34 | 46.34 | 2.266 |
| SliMeVR | 6 | 26.61 | 8.83 | 2.71 | 45.18 | 57.49 | 78.78 | N/A |
| GlobalPose | 6 | 27.29 | 9.91 | 37.11 | 69.94 | 31.29 | 49.33 | N/A |
| PNP | 6 | 27.74 | 16.29 | 21.40 | 45.14 | 34.13 | 51.17 | 1.664 |
| IMUCoCo | 6 | 34.15 | 11.91 | 39.64 | 24.54 | 72.19 | 70.84 | 0.754 |
| WheelPoser | **4** | 34.95 | 24.09 | 30.48 | 45.18 | 51.48 | 79.62 | N/A |

### 3.1 精度排序与传感器效率

TransPose 和 PIP 的全身旋转误差几乎相同，差值仅 0.46°。PIP 的平移误差最低，说明其根运动估计在该 DIP 设置下更稳定。MobilePoser 比 TransPose 高 1.04°、比 PIP 高 0.58°，但减少了一个传感器，体现出较好的精度/传感器折中。以全身旋转误差计算，MobilePoser 每个传感器对应约 4.12°，TransPose 约 3.26°；这只是描述性指标，不能替代模型复杂度或功耗评测。

MobilePoser 在膝部误差（13.46°）上显著领先其他方法，表明下肢局部姿态恢复较强；但上臂和前臂误差仍高于 TransPose/PIP，说明减少头部或上肢观测后，远端上肢存在更强的不确定性。WheelPoser 的 4 传感器结果在全身、腰部和上肢均明显落后，符合其面向轮椅用户上半身场景的专用定位，不宜作为一般人体动作的直接竞争基线。

## 4. AMASS drift 结果

| 方法 | 传感器数 | 全身旋转 (°) | 腰部 (°) | 平移 (m) | benchmark 状态 |
|---|---:|---:|---:|---:|---|
| PIP | 6 | **13.27** | **5.67** | **1.005** | 失败标记，结果需复核 |
| TransPose | 6 | 19.35 | 8.48 | 1.421 | passed |
| MobilePoser | **5** | 20.07 | 10.38 | **12.766** | passed |
| GlobalPose | 6 | 21.94 | 9.66 | N/A | 失败标记，结果需复核 |
| SliMeVR | 6 | 27.94 | 11.44 | N/A | passed |
| WheelPoser | **4** | 33.58 | 22.23 | N/A | passed |
| IMUCoCo | 6 | 33.61 | 11.77 | 1.942 | passed |
| PNP | 6 | N/A | N/A | N/A | 失败，运行中断 |

### 4.1 长时序稳定性

在成功生成标准结果的方法中，PIP 的旋转误差最低，TransPose 与 MobilePoser 接近。MobilePoser 的全身误差从 DIP 的 20.62° 变为 drift 的 20.07°，但这不是性能改善的证据，因为两套件样本不同。更可靠的观察是：在 AMASS 长时序集合上，5 传感器 MobilePoser 没有出现明显的全身旋转崩溃。

平移方面，MobilePoser drift 的 12.766 m 约为 TransPose 的 9 倍、PIP 的 12.7 倍。这是一个实质性问题：姿态角度看起来可接受，并不意味着角色在世界坐标中的行走、起身、躺下或接触地面位置正确。后续研究应将根平移、接触约束和速度积分误差作为独立优化目标。

### 4.2 动作覆盖

drift 输出目录包含 AMASS 动作视频，动作标签体系包括 standing、walking、running、jumping、sitting、lying、crawling、dancing、interaction、sports 和 transitions。当前 `standard_metrics.json` 只给出全局平均，无法回答以下关键问题：

- lying、crawling 等非直立动作是否造成腰部误差显著上升；
- transitions 是否引起根平移突变或姿态时序延迟；
- interaction、sports 等快速上肢动作是否扩大前臂误差；
- 各方法是否实际处理了相同数量的每类动作。

因此，本轮报告只将动作清单视为覆盖范围证据，不把视频数量当作精度指标。

## 5. 运行成本与可部署性

`duration_seconds` 是端到端 benchmark 运行时间，包含数据读取、模型推理、可视化和视频写出，不能直接当作单帧延迟。仍可做如下工程观察：

| 方法 | DIP 时间 (s) | drift 时间 (s) | 备注 |
|---|---:|---:|---|
| MobilePoser | 157.7 | 7,497.7 | 5 传感器，运行成功 |
| TransPose | 157.6 | 10,490.0 | 运行成功 |
| WheelPoser | 162.3 | 7,504.7 | 4 传感器，运行成功 |
| IMUCoCo | 175.9 | 9,046.5 | 运行成功 |
| PIP | 1,072.6 | 失败 | DIP 端到端耗时较高 |
| PNP | 967.1 | 失败 | drift 长时间后中断 |
| GlobalPose | 773.8 | 失败 | drift 启动后快速失败 |
| SliMeVR | 1.3 | 8,895.4 | DIP 时间可能主要是轻量/非视频路径，不宜与神经网络直接比较 |

这些时间受硬件、批大小、视频渲染和 I/O 影响。要验证“移动端轻量化”，还必须补充参数量、峰值内存、单帧 CPU/GPU 延迟、功耗和模型文件大小。

## 6. 研究判断

围绕“少于六个 IMU、移动端推理、非人体工学动作可靠性”的研究目标，当前证据支持以下判断：

- **可行性**：5 传感器 MobilePoser 在 DIP 和 AMASS drift 上保持约 20° 的全身旋转误差，证明减少到 5 个传感器具有可行性。
- **尚未解决的问题**：MobilePoser 平移漂移过大；上肢远端误差仍高；动作类别级鲁棒性尚未量化。
- **基线选择**：PIP/TransPose 是旋转精度基线，MobilePoser 是低传感器基线，WheelPoser 是 4 传感器专用场景基线。SliMeVR、GlobalPose、IMUCoCo 和 PNP 的适用域或失败状态应在表格中明确标注，不能只按平均角度排名。
- **研究风险**：当前 drift 失败任务使方法间的可比性不完整；若直接据此宣称某方法全面优越，会夸大结论。

## 7. 建议的下一轮实验

1. 对所有方法统一保存按动作、按序列的 `rotation_deg`、`translation_m`、有效帧数和失败原因。
2. 对 MobilePoser 至少比较 4、5、6 传感器组合，并报告腰部、膝部、上肢和根平移的变化曲线。
3. 将 lying、crawling、transitions、interaction 和 sports 设为强制报告类别，给出均值、P90 和失败率。
4. 单独评测根平移：短时 RMSE、每秒漂移率、接触脚滑移和起身/躺下终点误差。
5. 增加移动端指标：参数量、FP16/INT8 模型大小、CPU 单帧延迟、峰值 RAM 和能耗。
6. 重新运行 GlobalPose、PIP、PNP 的 drift 任务并保留完整 stderr/stdout；在失败任务补齐前，最终论文表格应标记为“未完成”，而不是填入残留指标。

## 8. 可复现文件

- 标准指标：`benchmark_results/<method>/<suite>/standard_metrics.json`
- 原始数组：`benchmark_results/<method>/<suite>/standard_metrics.npz`
- 运行状态与命令：`benchmark_results/<method>/<suite>/benchmark_report.json`
- 动作分类清单：`code/base_mobileposer/data/classification_manifest.csv`

