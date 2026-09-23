# 进度存档：mobileposer 实时推理替换 SlimeVR→Unity 姿态驱动

最后更新：2026-09-22。写这份文档是为了下次接着做时不用重新翻聊天记录。技术细节/组件说明见同目录
`README.md`，这份文档偏"发生了什么、为什么这么改、现在到哪一步了"。

## 目标（没变过）

链路：`IMU → SlimeVR-Server(校准，代码不改) → Python 实时桥接 → mobileposer 推理 → Unity 可视化`。
全部跑在一台 PC 上。用户实际硬件是"小腿/脚/上臂/腰部"这套无头 5-IMU 方案，对应仓库里 6 个
`no_head_5imu_surface` 布局 checkpoint 之一（腰+另外4个部位的组合，动态识别，不写死）。

## 当前状态：代码已写完、已用真实数据交叉验证过关键数学环节，等用户实机测试

## 做了什么（按时间顺序，含踩过的坑）

### 1. 架构调研阶段
调研清楚了 SlimeVR-Server(Kotlin)、android/ 现有的 mobile_poser ONNX 打包、SlimeVR 的 SolarXR
WebSocket 协议(`ws://127.0.0.1:21110`)。确定了：不用改 SlimeVR-Server 代码，直接订阅它自带的
SolarXR feed 就能拿到逐 tracker 的已校准旋转+加速度+BodyPart 标签。

### 2. 第一版实现（有严重错误，已修正）
第一版实现犯了两个错：
- 世界系变换矩阵用错了（借用了 AMASS 数据处理里的 Z-up→Y-up 转换，但实时数据本身是 Y-up，不需要）。
- 想用"逐帧代数修正"的取巧方案处理加速度，**声称不需要标定**。

### 3. 用户追问"这样对吗" → 发现仓库里已有权威参考
用户提醒：仓库里 `mobileposer/test_ours_surface.py` 已经用真实录制数据(`data/raw/ours`)跑过
这套 checkpoint 的推理，并且做了定量验证（跟同次录制的 Unity Avatar 回放对比，算关节位置差异
厘米数，报告存在 report.json/README.md 里）。这是比我自己推导权威得多的参照物。

对比后发现的问题：
- `SERVER_TO_SMPL` 应该是 `diag(-1,1,-1)`（Server→VMC 反射Z，VMC→SMPL 反射X），不是我用的 AMASS
  那套矩阵。
- 加速度的"逐帧代数修正不需要标定"这个结论是错的——用真实数据交叉验证，拟合出的隐含 yaw 和
  `test_ours_surface.py` 拟合出的结果对不上（有的差 100°+）。

### 4. 去 SlimeVR-Server 源码里核实（三路并行调研）
确认了根本原因：`rotationReferenceAdjusted(t) = LeftConst @ rotation(t) @ RightConst`
（`TrackerResetsHandler.kt:180-254`），`RightConst` 由 4 个四元数相乘而成，SolarXR 协议只暴露其中
1 个(`mountingOrientation`)，另外 3 个是私有字段——**逐帧代数修正在数学上不可能**，之前的"取巧
方案"是错的。同时也确认了 `tracker.json`（另一个录制产物）是 SlimeVR 解算完的 VMC 输出，不是原始
传感器数据，不能用来做标定；`raw-imu.json`/SolarXR 这条路才是对的。

### 5. 重写 + 用真实数据交叉验证（当前版本）
- 姿态：`bone_orientation()`（无需标定，手臂类传感器固定 ±90° Z 偏置 + `SERVER_TO_SMPL` 转换）——
  跟 `test_ours_surface.py` 逐帧比对，误差 3e-8（浮点精度级别，一致）。
- 加速度：改成"开局标定"（`fit_heading()`/`calibrate_heading()`，需要一段有转动的数据，拟合每个
  tracker 固定的偏航修正）——用真实数据实测：只用前 5 秒(150帧)拟合，跟用全部 1863 帧拟合相比，
  偏航角只差 0.14°~1.14°，最终加速度数值差 0.0002~0.0039 m/s²，可以接受。标定完之后每帧只是一次
  固定矩阵乘法，不影响实时性。
- 6 个真实 checkpoint（用户已经从服务器同步到本地 `checkpoints/no_head_5imu_surface/`）全部验证
  能正常加载（539.5万参数/个）。
- 确认可以纯 CPU 推理，不需要 GPU（Android 那边本来就是 CPU-only ONNX 跑的，模型只有 5.4M 参数，
  PC CPU 跑 30Hz 实时完全没压力）。

## 现在的文件清单

```
mobileposer/realtime/
  solarxr_client.py   订阅 SolarXR，拿 rotation(未校准)/rotation_reference_adjusted(已校准)/
                       linear_acceleration/BodyPart/Status
  layout.py            BodyPart → mobileposer 标签映射 + 6 选 1 布局精确匹配
  calibration.py       bone_orientation()无状态姿态转换 + fit_heading()/calibrate_heading()
                       开局加速度标定 + apply_heading()逐帧应用
  rotmat.py            旋转矩阵↔四元数 + SMPL↔Unity 坐标系转换(diag(-1,1,1)，注意跟
                       calibration.py 里的 SERVER_TO_SMPL=diag(-1,1,-1) 是两个不同的转换)
  infer.py             45帧滑窗 + 复用 mobile_export.MobilePose 推理
  stream_out.py         本地 WebSocket 广播给 Unity（JSON，扁平数组）
  run.py               入口：等布局识别 → 开局标定(提示用户转动) → 持续推流
  probe_solarxr.py      诊断脚本，光打印 SolarXR 数据，不跑推理
  solarxr_schema/        flatc schema 源文件
  README.md             组件/架构参考（技术细节）
  PROGRESS.md            本文档

solarxr_protocol/        仓库根目录，flatc 生成的 Python 绑定（已手工修过 flatc --gen-all 的
                         import 缺失 bug，不要用 flatc 重新生成后直接覆盖，会带回这个 bug）

IMUTrack-for-Spine/Assets/slimeVR/Scripts/MobilePoserPoseSource.cs
                         Unity 接收脚本，挂在头像 Animator 上，用"相对静息姿势增量"驱动骨骼
```

## 运行流程（给下次接着做时直接照做）

### 前置条件
- `checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth` 已同步到本地（已确认 6 个都在）。
- Python 环境要有 `torch numpy scipy tqdm flatbuffers websockets`（CPU 版 torch 即可，不需要
  GPU）。开发用的环境在 `E:\dyh\conda_envs\mobileposer-realtime`。
- SlimeVR-Server 桌面版运行中，IMU 已连接，已做过 SlimeVR 自己的"位姿校准"。

### 第一步：只看数据对不对
```bash
cd E:\dyh\MotionRecover\code\base_mobileposer
<python> -m mobileposer.realtime.probe_solarxr
```
看 5 个 tracker 的 BodyPart/状态/旋转/加速度是否实时刷新、晃动对应传感器数值是否明显变化。

### 第二步：跑完整链路
```bash
<python> -m mobileposer.realtime.run
```
1. 打印识别出的布局名（当前绑定的 5 个 tracker 要精确匹配某个已训练布局，不匹配会一直等）。
2. 提示"转动/摆动几秒"——**必须真的动**（扭腰、抬手臂腿），约 5 秒/150 帧，动得不够会自动重试。
3. 打印每个 tracker 的拟合结果（`yaw=... residual_p95=...`），`residual_p95` 正常应该是个位数度数
   （参考：真实数据测出来大多 0.1°~1°）。
4. 打印 "Streaming live pose..."，开始持续向 `ws://127.0.0.1:21200` 推流。

### 第三步：接 Unity
1. 打开 `IMUTrack-for-Spine`，找到驱动 `mesh_edit1.vrm` 的场景。
2. 把 `MobilePoserPoseSource`（`Assets/slimeVR/Scripts/MobilePoserPoseSource.cs`）挂到头像
   `Animator` 所在物体上（或手动把 `animator` 字段指过去）。
3. 运行场景，自动连 `ws://127.0.0.1:21200`。
4. 做动作看头像是否跟着动、方向对不对、有没有关节拧巴。

## 还没做 / 已知风险（下次接着查这些）

1. **Unity 骨骼重定向没法在这个环境里验证**：`MobilePoserPoseSource.cs` 用"相对静息姿势增量"
   （`t.localRotation = rest[i] * received`），这是标准轻量做法，但不保证每个关节轴系都对得上
   （肩、髋最容易出问题）。需要实机跑起来肉眼看，如果某个关节拧了，要单独查那个关节的轴系差异，
   不用整体推翻方案。
2. **`LeftConst`（拟合出的 yaw）长会话可能漂移**：如果 SlimeVR 开了 drift compensation 或
   StayAligned，这个值不是绝对常数，会缓慢漂移。当前实现只在开局标定一次，没做定期重新标定。
   如果长时间跑下来发现姿态整体缓慢跑偏，这是嫌疑点。
3. **没有热切换（原计划 M5）**：tracker 绑定变了要重启 `run.py`，没做自动重新识别布局。
4. **加速度只用了 Poser 分支**：当前只训了/用了姿态(旋转)推理，没有全局平移(Joints/Velocity
   分支)，Unity 里的头像不会有根节点位移，只有原地姿态动画。如果之后需要位移，要看
   `no_head_5imu_surface` 有没有对应的 Joints/Velocity checkpoint，或者需要额外训练。

## 排查思路（如果实测出问题）

- **完全连不上/没数据**：先用 `probe_solarxr.py` 单独排查，跟 Unity/推理无关。
- **标定总是失败（UnobservableHeadingError）**：转动幅度不够，或者转动方式太单一（比如只绕一个轴转，
  参考 `calibration.py` 里 `fit_heading` 的实现，单轴转动在数学上是不可观测的，需要多轴混合运动，
  比如扭腰+摆臂）。
- **姿态整体方向感觉对，但某个关节拧巴**：大概率是 `MobilePoserPoseSource.cs` 的轴系重定向问题
  （风险点 1），查那个具体关节。
- **姿态整体方向就是错的（比如左右反了、上下颠倒）**：先怀疑 `SERVER_TO_SMPL`/`UNITY_SMPL_BASIS`
  这两个坐标转换矩阵，虽然已经跟 `test_ours_surface.py` 交叉验证过姿态部分（3e-8 误差），但那次验证
  用的是离线录制数据格式，实时 SolarXR feed 字段解析(`solarxr_client.py`)本身如果有 bug，交叉验证
  是测不出来的——先用 `probe_solarxr.py` 确认原始四元数数值本身是否合理。
