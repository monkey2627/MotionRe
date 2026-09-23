# mobileposer realtime bridge

Replaces "SlimeVR solves the pose and drives the Unity avatar" with
"mobileposer solves the pose and drives the Unity avatar", while leaving
SlimeVR-Server's own IMU ingest/calibration/tracker-binding and its VMC/OSC
output completely untouched. Everything below runs on one PC, alongside
SlimeVR-Server and Unity. See `C:\Users\admin\.claude\plans\misty-imagining-quasar.md`
for the original design rationale; this file reflects the corrected
implementation after cross-validating against real data (see "Validation"
below) -- a first draft of this module got the acceleration conversion wrong,
and that mistake plus the fix are documented here so it isn't repeated.

## Pipeline

```
IMU -> SlimeVR-Server(桌面版，代码不改)
    -> solarxr_client.py   (订阅 ws://127.0.0.1:21110, 拿 rotation + rotation_reference_adjusted
                             + linear_acceleration + BodyPart, 不改 SlimeVR-Server 代码)
    -> layout.py            (当前在线 tracker 集合精确匹配 no_head_layouts 的 6 个布局之一)
    -> [启动一次] calibration.py::calibrate_heading
                             (要求用户转动几秒，拟合每个 tracker 的固定偏航修正 -- 这一步不可省略，见下)
    -> infer.py             (逐帧: calibration.py::bone_orientation 转姿态[无需标定] +
                             heading.apply 转加速度[用标定好的固定矩阵] -> 45帧滑窗 ->
                             mobile_export.MobilePose 推理)
    -> stream_out.py        (本地 WebSocket 广播 SMPL 24 关节四元数, Unity 惯例坐标系)
    -> Unity: Assets/slimeVR/Scripts/MobilePoserPoseSource.cs (驱动 Humanoid Animator)
```

## 关键结论(已用真实数据交叉验证，不是纯推导)

1. **姿态不需要标定**。`rotation_reference_adjusted` 已经包含 SlimeVR 自己的位姿校准，只需要
   对手臂类传感器(`lw`/`lu`/`rw`/`ru`)叠加一个固定的 ±90° 绕 Z 偏置(补偿 SlimeVR 内部手臂参考
   系跟 SMPL T-pose 参考系的差异)，再做 `SERVER_TO_SMPL = diag(-1,1,-1)` 坐标转换。这部分逐帧
   跟 `mobileposer/test_ours_surface.py`（已经用真实数据做过定量验证的脚本）比对，误差在
   3e-8 量级(浮点精度)，完全一致。

2. **加速度必须要标定，躲不掉，且已经用 SlimeVR-Server Kotlin 源码逐行确认过原因**：
   - SlimeVR 内部算 `linear_acceleration` 用的是**未校准的** `rotation`，不是 `rotation_reference_adjusted`
     (`Tracker.kt:456-460` → `TrackerResetsHandler.kt:196`)。
   - `rotation_reference_adjusted(t) = LeftConst @ rotation(t) @ RightConst`，其中 `RightConst`
     由 4 个四元数相乘而成(`TrackerResetsHandler.kt:180-254`)，SolarXR 协议里只暴露了其中 1 个
     (`mountingOrientation`)，另外 3 个是私有字段，**协议里永远拿不到**——所以任何"只用当前帧
     数据做代数修正"的方案在数学上都不可能，之前设计的"逐帧代数修正，不用标定"是错的（已经在
     真实数据上验证出明显不一致，最多差 260°）。
   - 唯一可行的办法是 `test_ours_surface.py` 用的"用一段有转动的数据，通过相对第0帧的相对旋转
     轨迹去拟合"，这样能让未知的 `RightConst` 在代数上抵消掉，解出可用的部分(`LeftConst`，约等于
     一个纯偏航角)。**静止不转是拟合不出来的**（数学上不可观测，代码里有专门检查会报错）。
   - 好消息：这个拟合只需要一小段数据，不需要整场。用真实录制数据实测：只用前 5 秒(150 帧)拟合，
     跟用全程 1863 帧拟合的结果比，偏航角差 0.14°~1.14°，最终加速度数值差 0.0002~0.0039 m/s²
     ——几乎可以忽略。拟合完之后，之后每一帧的应用就是一个固定矩阵乘法，不影响实时性。

3. **加速度单位**：SolarXR 实时 feed 给的 `linear_acceleration` 已经是 m/s²（不是 G），不需要
   乘 9.80665——这点跟离线脚本处理录制文件(`accelerationG` 字段，G 单位)不一样，注意别搞混。

## 组件

- `solarxr_client.py`：订阅 `rotation`(未校准)、`rotation_reference_adjusted`(已校准)、
  `linear_acceleration`、`Info.BodyPart`、`Status`。
- `layout.py`：BodyPart → mobileposer 标签映射 + 布局精确匹配(不匹配就报错，不猜)。
- `calibration.py`：
  - `bone_orientation()`：无状态，逐帧姿态转换。
  - `fit_heading()` / `calibrate_heading()`：启动时跑一次，需要一段转动数据，拟合每个 tracker
    的固定偏航修正矩阵；转动不够会抛 `UnobservableHeadingError`。
  - `apply_heading()`：标定完成后，逐帧加速度转换，纯矩阵乘法。
- `infer.py`：45 帧滑窗 + 复用 `mobile_export.MobilePose`(已知跟真实 checkpoint 精度对齐)。
- `stream_out.py` / `rotmat.py`：广播给 Unity，含 SMPL↔Unity 坐标系转换(`diag(-1,1,1)`，跟
  `SERVER_TO_SMPL` 是两个不同的转换，分别用在输入/输出两端，不要混用)。
- `run.py`：串起来的入口，含启动标定阶段的用户引导("转动几秒")。

## Before running

1. **Checkpoint**：`checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth` 只在训练服务器
   上，需要同步到本地对应路径，或用 `--checkpoint-root` 指定。
2. **Python 依赖**：`torch`、`numpy`、`scipy`(解 SMPL pickle、拟合 yaw 都要用)、`tqdm`、
   `flatbuffers`、`websockets`。开发时建的 `mobileposer-realtime` conda 环境在
   `E:\dyh\conda_envs\mobileposer-realtime`，按需自建。
3. **SolarXR 绑定**：生成的 flatbuffers Python 包在仓库根目录 `solarxr_protocol/`，由
   `mobileposer/realtime/solarxr_schema/all.fbs` 生成：
   ```
   flatc --python --gen-all --gen-object-api -o <out> mobileposer/realtime/solarxr_schema/all.fbs
   ```
   **注意**：flatc 25.12.19 的 `--gen-all` 在 Python 输出上有真实 bug（跨文件引用漏 import），
   已经在仓库里手工修过；如果重新生成需要重新处理，报错信息是 `NameError`。
4. **SlimeVR-Server** 运行中，IMU 已连接并绑定到这个 checkpoint family 认识的 TrackerPosition
   (腰 + 小臂/上臂/大腿/小腿/脚 的子集，见 `layout.py::BODY_PART_TO_LABEL`)。

## Running

```
python -m mobileposer.realtime.probe_solarxr        # 先看实时 feed 数据对不对
python -m mobileposer.realtime.run                   # 完整链路，含启动标定
```

`run.py` 流程：
1. 等待当前绑定的 tracker 精确匹配某个已训练布局。
2. 提示"转动/摆动几秒"，收集标定窗口数据，拟合每个 tracker 的偏航修正(转不够会自动重试)。
3. 加载对应 checkpoint，开始向 `ws://127.0.0.1:21200` 实时推流。

Unity 里把 `MobilePoserPoseSource` 挂到头像 `Animator` 所在的 GameObject 上，运行即可。

## 仍未验证/已知风险

- **Unity 骨骼重定向**：`MobilePoserPoseSource.cs` 用"相对静息姿势的增量"来套用旋转，这是标准
  轻量做法，但没法排除个别关节(肩、髋常见)轴系对不上的可能，需要在 Unity 里实测看。
- **LeftConst 的漂移**：如果 SlimeVR 开了 drift compensation 或 StayAligned，`LeftConst`(拟合出
  的偏航)会随会话时间缓慢漂移，不是绝对常数。当前实现只标定一次，长时间会话可能需要定期重新标定
  (未实现)。
- 没有热切换(plan M5)：tracker 集合变了要重启 `run.py`。
