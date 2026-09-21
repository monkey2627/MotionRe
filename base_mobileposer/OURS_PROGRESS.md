# ours/1 数据适配、测试与微调进度

更新时间：2026-09-18

## 目标

将 `data/raw/ours/1` 这次采集的 IMU 数据，按照六种 no-head surface-IMU 配置输入对应 MobilePoser checkpoint，检查预测效果，并验证同一动作上的微调 smoke test。

采集包路径：`data/raw/ours/1`。其中包含 `motion.json`、`tracker.json`、`raw-imu.json`、FBX 和 `manifest.json`。

## 已理解的原始链路

原始链路记录在 [`IMU到动作导出全链路.md`](IMUTrack-for-Spine/IMU到动作导出全链路.md)：

```text
IMU 固件融合 → SlimeVR Server 校准/人体解算 → Unity Avatar
→ motion.json / FBX / tracker.json / raw-imu.json
```

`raw-imu.json` 与 `motion.json` 是并行产物，不能把 Avatar 姿态当成原始 IMU。当前 `raw-imu.json` 为 SolarXR 记录的线性加速度和多种四元数。

surface IMU 的训练定义位于 [`surface_imu.py`](mobileposer/surface_imu.py)、[`no_head_surface_imu.json`](mobileposer/configs/no_head_surface_imu.json) 和 [`no_head_layouts.py`](mobileposer/no_head_layouts.py)：固定 SMPL 表面三角面、重心权重和骨段方向，再生成加速度与方向监督。

## 传感器与布局映射

采集中的实体角色为：

|模型槽位|raw-imu trackerRole|说明|
|---|---|---|
|`lw`|`LEFT_LOWER_ARM`|左前臂；实际不是手部|
|`rw`|`RIGHT_LOWER_ARM`|右前臂；实际不是手部|
|`lu`|`LEFT_UPPER_ARM`|左上臂|
|`ru`|`RIGHT_UPPER_ARM`|右上臂|
|`lt` / `rt`|`LEFT/RIGHT_UPPER_LEG`|左右大腿|
|`ls` / `rs`|`LEFT/RIGHT_LOWER_LEG`|左右小腿|
|`waist`|`WAIST`|腰部|
|`lf` / `rf`|缺失|采集没有左右脚实体 IMU|

因此五组布局可测试：

- `wrists_thighs_waist`
- `wrists_shanks_waist`
- `upperarms_thighs_waist`
- `wrists_upperarms_waist`
- `legs_waist`

`wrists_feet_waist` 未测试。没有用虚拟脚 tracker 或零值补齐。

## 输入适配实现

主代码为 [`test_ours_surface.py`](mobileposer/test_ours_surface.py)。它完成：

1. 校验 `manifest.json` 中 raw-imu SHA256；
2. 验证采样时间单调、无明显丢帧；
3. 按 `trackerRole` 严格绑定传感器身份；
4. 使用 `rotationReferenceAdjusted` 方向；
5. 对 SlimeVR 线性加速度的恒定 yaw 偏差做传感器自身的离线相对旋转拟合，不读取动作姿态；
6. 加速度从 G 转成 `m/s²`；
7. 按时间戳重采样到 30 FPS；
8. 转换 VMC/Unity 与 SMPL 的 Y-up 坐标基；
9. 生成每个 layout 的 `(T, 5, 3)` 加速度和 `(T, 5, 3, 3)` 方向；
10. 检查 surface checkpoint 的 attachment hash；
11. 使用对应 checkpoint 推理并保存预测。

运行命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python \
-m mobileposer.test_ours_surface
```

## 第一次参考动作问题及修正

最初使用 Blender 从 FBX 导出的骨架作为视频蓝色参考。后来发现它与用户在 `IMUTrack-for-Spine/3/IMUTrack.exe` 中播放 `motion.json` 的结果不一致，尤其腿脚出现异常翻折。因此：

- `results/ours1_surface_realimu` 中旧的 FBX 参考视频和数值已标记作废；
- 当前正式参考改为 Unity `HumanPoseHandler.SetHumanPose` 播放 `motion.json`；
- 使用项目中的 `mesh_edit1.vrm` 和 [`HumanPoseJointExporter.cs`](IMUTrack-for-Spine/Assets/Editor/HumanPoseJointExporter.cs)；
- 参考按 `motion.json` 实际时间戳与 IMU 对齐；
- 当前预测输入没有使用 FBX 或 Avatar 姿态。

正式参考结果目录：[`results/ours1_surface_motionjson`](results/ours1_surface_motionjson)

其中：

- `reference/ours1.humanpose_joints.json`：Unity HumanPose 导出；
- `reference/joints_y_up.npy`：转换到 SMPL Y-up 的参考关节；
- `reference_correction.png`：旧 FBX 与新 HumanPose 参考的异常帧对比；
- `report.json`：传感器、时间戳、参考和 checkpoint 校验信息；
- `inputs/<layout>/ours1.pt`：五组实体 IMU 输入；
- `<layout>/prediction.pt`：预测 pose、joint、tran、contact；
- `<layout>/comparison.mp4`：蓝色 HumanPose 参考、红色预测。

## 初始推理结果

旧版结果不再使用。修正后的 HumanPose 参考下，根节点对齐后的平均 Avatar 关节差异约为：

|布局|平均差异|
|---|---:|
|`wrists_thighs_waist`|14.96 cm|
|`wrists_shanks_waist`|15.18 cm|
|`upperarms_thighs_waist`|13.79 cm|
|`wrists_upperarms_waist`|13.78 cm|
|`legs_waist`|15.23 cm|

该指标只是两条算法链路之间的一致性，不是独立动捕真值下的重建误差。

## 同动作微调 smoke test

用户目标是验证：在同一个 `ours/1` 动作上微调后，再预测该动作是否改善。为此新增 [`smoke_finetune_ours.py`](mobileposer/smoke_finetune_ours.py)。

实验定义：

- 输入：`results/ours1_surface_motionjson/inputs`；
- 标签：当前 Unity HumanPose 经 SMPL rest-pose 校准后的伪标签；
- 初始化：对应 `checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth`；
- 可训练模块：`joints` 和 `poser`；
- 冻结模块：`velocity` 和 `foot_contact`；
- 训练：整段 825 帧，120 次优化更新，关闭外部 dropout；
- 评估：同一 825 帧，没有独立验证集；
- 原始 checkpoint 未覆盖修改。

运行命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python \
-m mobileposer.smoke_finetune_ours \
--output results/ours1_overfit_smoke_new
```

结果目录：[`results/ours1_overfit_smoke`](results/ours1_overfit_smoke)

|布局|SMPL 伪标签 MPJPE 前 → 后|Avatar 差异前 → 后|
|---|---:|---:|
|`wrists_thighs_waist`|13.03 → 0.89 cm|14.96 → 5.25 cm|
|`wrists_shanks_waist`|13.75 → 0.96 cm|15.18 → 5.27 cm|
|`upperarms_thighs_waist`|11.95 → 0.85 cm|13.79 → 5.24 cm|
|`wrists_upperarms_waist`|11.28 → 0.84 cm|13.78 → 5.24 cm|
|`legs_waist`|14.77 → 0.97 cm|15.23 → 5.28 cm|

结果说明微调代码、标签读取、checkpoint 加载和同动作过拟合链路均已接通。它不能说明泛化能力，也不能证明真实 SMPL 精度，因为标签是同包 HumanPose 转换得到的伪标签。

## 代码和结果索引

|用途|位置|
|---|---|
|原始数据|`data/raw/ours/1`|
|链路说明|`IMUTrack-for-Spine/IMU到动作导出全链路.md`|
|surface 选点|`mobileposer/surface_imu.py`|
|surface 配置|`mobileposer/configs/no_head_surface_imu.json`|
|布局和训练数据定义|`mobileposer/no_head_layouts.py`|
|实体 IMU 适配与推理|`mobileposer/test_ours_surface.py`|
|同动作微调|`mobileposer/smoke_finetune_ours.py`|
|HumanPose 导出|`IMUTrack-for-Spine/Assets/Editor/HumanPoseJointExporter.cs`|
|初始推理结果|`results/ours1_surface_motionjson`|
|微调结果|`results/ours1_overfit_smoke`|
|surface checkpoint|`checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth`|
|适配回归测试|`tests/test_ours_surface_adapter.py`|

## 已完成的验证

- 五组 layout 输入均为 825 帧，方向矩阵正交且行列式为 1；
- 五组预测输出无 NaN/Inf；
- 五组 before/after 视频均完整解码；
- 原始 checkpoint SHA256 在微调前后保持不变；
- `joints`、`poser` 之外的冻结模块权重未改变；
- 5 项适配回归测试通过。

## 当前限制和下一步

当前限制：只有一个约 27.5 秒动作、没有实体脚部 IMU、没有独立 SMPL 真值、HumanPose→SMPL 是伪标签、微调测试训练和测试使用同一序列。

下一步可以按优先级做：

1. 收集更多动作，保持同一传感器佩戴和 layout；
2. 用动作划分做 train/validation，而不是随机切同一动作；
3. 记录多段不同速度、方向和上下肢动作；
4. 若要测试脚部 layout，需要实际左右脚 IMU；
5. 有独立 SMPL/动捕标签后再报告正式精度；
6. 将当前 smoke-test 入口扩展成多序列微调和 held-out 评估入口。
