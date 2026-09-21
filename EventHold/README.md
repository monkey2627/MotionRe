# EventHold：无头部五 IMU 姿态记忆研究

当前实现阶段：G0 数据／输入审计、独立适配器及因果 GRU 管线试跑。
目标物理传感器为骨盆、双前臂、双小腿。没有复用旧统一桥接，也没有修改既有方法源码。

**事件记忆模型尚未实现；已完成两轮管线试跑和 30 轮 DIP-only 初始 B0；这些不构成强基线或论文结果。**

## 可运行内容

在本目录使用已有 mobileposer 环境：

```bash
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m unittest discover -s tests -v
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m eventhold.audit
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m eventhold.verify_native
```

基线管线试跑（使用新输出目录，避免覆盖权重）：

```bash
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m eventhold.train_baseline \
  --output reports/b0_new_pilot --device cuda:0 --epochs 2 \
  --max-per-subject 1 --max-frames 3600
```

GPU 编号可自行指定；不需要运行管理员安装或修改现有环境。这里仅使用旋转监督、未做 AMASS 预训练或 FK 损失，输出为姿态，没有估计根轨迹。

## 数据规则

- 直接读取原始 DIP，目标索引 `[2,7,8,11,12]`，不使用旧的双大腿缓存。
- 保留原始 60 Hz；此决定替代草案里本模型的 30 Hz 初值，避免未经验证的重采样。原生外部方法各自核对 dt。
- s_01–s_06 训练，s_07–s_08 验证，s_09–s_10 锁定测试。审计可以检查测试形状和缺失情况，但不挖掘测试姿态保持段或用于模型选择。
- 缺失输入仅前向填补、显式掩码；真值姿态不进入适配器或模型初始化。
- DIP 无实测根轨迹，不给其零占位平移打分。
- 自动挖掘到的是低角速度候选，不自动等于坐姿、蹲姿或真实长保持。

## 独立方法适配

`contracts/` 保存六个方法的契约、来源哈希、已核查项与未完成项。
PNP 的世界系线加速度和角速度、DynaIP 的根系特征分别转换。
`verify_native` 直接执行本地原生源码中的预处理方程进行对照，不使用旧桥接充当参考。

通过特征对照 **不代表** 已通过 checkpoint 逐帧推理对照，更不代表六传感器权重可用于五传感器。准入状态保持明确分层。

## 结果入口

- `reports/g0/data_audit.json`：原始数据清单、缺失率、源代码哈希。
- `reports/g0/sequence_manifest.csv`：逐序列来源、划分、时长。
- `reports/g0/hold_candidates.csv` / `review_queue.csv`：待核查候选。
- `reports/native_preprocess_parity.json`：原生预处理对照。
- `reports/b0_pipeline_pilot/`：小规模训练记录、最终权重与验证预测。
- `reports/G0进展.md`：本阶段发现与后续准入条件。

研究计划见 [执行计划](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/执行计划.md)。

## 完整初始 B0 诊断与 PNP 推理对照

```bash
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m eventhold.diagnose_baseline \
  --run reports/b0_dip_full_train_seed0 --output reports/b0_dip_full_train_seed0/diagnosis

PYTHONPATH="$PWD/runtime/pip37:$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/duanyuhan/SoftWare/miniconda3/envs/pip/bin/python -m eventhold.verify_pnp_forward \
  --output reports/pnp_forward_parity.json
```

PNP 对照已经通过原生六传感器 checkpoint 推理；五传感器准入仍未完成。完整 B0 使用最终轮权重，结果和数据缺口见阶段报告。

## 保留/写入离线诊断（不是模型排名）

```bash
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m eventhold.diagnose_update_choices \
  --run reports/b0_dip_full_train_seed0 \
  --output reports/b0_dip_full_train_seed0/update_choice_diagnostic
```

所有策略依赖 GT 挖掘的候选边界；oracle 额外使用参考姿态选择，不能用于部署成绩。当前只有短候选，无法检验长保持或退出延迟。结果与近邻查新见[诊断记录](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/近邻查新与最小诊断结果.md)。全套 28 项测试通过。

## 姿态歧义与历史信息诊断

`eventhold.diagnose_ambiguity` 在验证集同一序列内进行测量匹配，再评分参考髋膝/脊柱角差。当前条件和缺失处理敏感性分别固定在 `contracts/ambiguity_diagnostic_v1.json`、`contracts/ambiguity_diagnostic_sparse_validity.json`。

```bash
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m eventhold.diagnose_ambiguity \
  --config contracts/ambiguity_diagnostic_sparse_validity.json --output reports/ambiguity_new_run
```

图库使用整段验证序列，属于离线诊断，不能报告为因果部署效果。已完成输出见 `reports/ambiguity_all_pairs/`；原连续有效历史分析见 `reports/ambiguity_v1/`。当前全套 33 项测试通过。研究结论见[诊断报告](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/姿态歧义与转移证据诊断.md)。

## Natural Motion 真实长保持数据

独立流式读取器 `eventhold.natural_motion` 保留 MVNX 导出 240 Hz，按名称选骨盆、双前臂、双小腿；未用参考骨段姿态标定传感器。P5/P13 均为探索数据，不是锁定测试。

```bash
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m eventhold.natural_motion \
  --archive runtime/natural_motion/P5_Day_1.zip \
  --output reports/g0/natural_motion/P5_new_run --cache runtime/natural_motion/P5_new_run
```

P5 有真实长坐姿形态候选，但未观察到完整退出，且模型输入标定未通过准入。结果、参考曲线与骨架图见[长保持核查报告](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/NaturalMotion长保持数据核查.md)。当前全套 36 项测试通过。图是 Xsens 参考，不是本模型预测。

## 显式安装标定诊断

`eventhold.mounting` 区分已知姿态与参考辅助前缀标定，默认拒绝将后者用于部署接口。P5 的前 2.5 秒参考辅助固定偏移诊断已完成，原始缓存未改变。

```bash
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m eventhold.audit_mounting \
  --cache runtime/natural_motion/P5_Day_1/P5_Day_1_1.npz \
  --output reports/g0/natural_motion/P5_Day_1/mounting_diagnostic
```

当前全套 40 项测试通过。此结果不是模型性能，部署标定和模型评价准入仍未完成。详见[标定核查报告](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/标定协议与原始输入核查.md)。

## P5 完整序列 B0 诊断（2026-09-21）

已实际运行参考辅助标定、DIP-only B0 的 68017 帧持续推理；输出在 `reports/g0/natural_motion/P5_Day_1/b0_reference_assisted_v1/`。仅比较四个髋膝局部旋转代理，不宣称完成全身 SMPL 重定向。C1 初始角差已很大，未建立遗忘证据。

当前全套 44 项测试通过。具体延迟、局限、复现命令及下一步见[诊断报告](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/NaturalMotion五IMU基线诊断.md)。部署主实验仍未准入。

## AMASS 覆盖核查（2026-09-21）

`audit_amass_coverage` 完成 9336 条原始动作筛查；`confirm_amass_candidates` 对 65 个阳性文件完成原始帧率复查。8 段坐姿形态超过 20 秒，但严格低速坐姿最长 7.54 秒。旧 MobilePoser 大腿/头部缓存不可复用。未训练新模型。当前全套 46 项测试通过。

详见[训练覆盖报告](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/AMASS训练覆盖与合成核查.md)与 `contracts/amass_synthesis_draft.json`。

## 五 IMU 合成小批验证

当前有效划分 `contracts/amass_split_v3/`，样例 `runtime/amass_pilot_v3/`。v1/v2 均已标记被替代，禁止混入训练对照。六条受控平均男性体型样例合成完成；51 项测试通过，尚未训练新模型。参见[详细记录](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/五IMU合成试运行与划分.md)。

## 五 IMU 合成试运行（2026-09-21）

已生成 `reports/g0/five_imu_synthetic_pilot/`：8 条候选序列、60 Hz、骨盆/双前臂/双小腿，无头部。合成加速度为离线中心差分，边界 mask 已记录，不能视为真实 IMU。当前全套 52 项测试通过。

详情见[合成试运行记录](/home/duanyuhan/dyh/motion/Research/proposal_5imu_no_head/event_memory_plan/五IMU合成试运行.md)。

## 合成五 IMU Smoke Test

8 条合成序列已贯通 66 维 B0 接口；首尾边界 mask、根旋转和连续状态分块一致性通过。结果位于 `reports/g0/five_imu_synthetic_smoke/summary.json`，全套 53 项测试通过。该结果不是性能评估。

## 合成来源划分

当前 8 条合成序列已按 `subject_key` 冻结探索性 train/holdout 划分，清单位于 `reports/g0/five_imu_synthetic_split/split.json`。这不是最终测试协议。

## M1 区域事件写入原型

`eventhold.region_event_model.RegionEventWrite` 已实现下肢/骨盆与前臂独立写入门；测试覆盖未来扰动和连续状态分块一致性。当前全套 56 项测试通过。M1 尚未训练或报告性能。
