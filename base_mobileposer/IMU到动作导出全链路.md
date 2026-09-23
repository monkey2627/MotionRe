# IMU → SlimeVR → Unity → 动作包：完整数据链路

## 总览

```text
IMU 芯片原始采样
  ↓ Tracker 固件姿态融合
Tracker 数据包（旋转四元数、加速度、状态）
  ↓ UDP
SlimeVR Server
  ↓ 校准、佩戴方向修正、Yaw/漂移补偿、滤波
Tracker 有效姿态
  ↓ HumanSkeleton
人体骨骼姿态
  ↓ VMC OSC
Unity / EVMC4U Avatar
  ↓ PoseFbxRecorder
pose.json
  ↓ RuntimeMotionPackageExporter
FBX + motion.json + tracker.json + raw-imu.json + manifest.json
```

要区分三种数据：

- **Tracker 数据**：Tracker 发给 SlimeVR 的协议数据；IMU Tracker 的旋转通常已经由固件融合完成。
- **Raw IMU 记录**：Unity 通过 SolarXR 旁路保存的传感器数据和融合四元数，用于回放、诊断或重建。
- **Motion 数据**：Unity Avatar 的最终人体骨骼姿态，来源是 SlimeVR 解算后通过 VMC 驱动的模型。

`raw-imu.json` 不会在导出时重新计算成 `motion.json`，两者是同一次录制中的并行产物。

## 1. Tracker：IMU 原始数据变成姿态

Tracker 内的陀螺仪、加速度计和（视硬件而定）磁力计由固件融合成旋转四元数，再通过 SlimeVR Tracker UDP 协议发送。SlimeVR Server 的网络输入是 Tracker 协议包，而不是裸的三轴陀螺仪时间序列。协议通常还包含加速度、电量、温度、设备状态、传感器类型和采样频率。

因此，研究“陀螺仪/加速度计如何融合成四元数”需要查看对应 Tracker 固件；Server 主要消费融合后的 Rotation。

## 2. SlimeVR Server 接收 Tracker

Server 目录：

```text
/home/duanyuhan/dyh/motion/MotionRe/slimevr/SlimeVR-Server
```

入口：`server/core/src/main/java/dev/slimevr/tracking/trackers/udp/TrackersUDPServer.kt`。

`processPacket(...)` 对旋转和加速度包执行：

```kotlin
tracker.setRotation(rot)
tracker.setAcceleration(packet.acceleration)
tracker.dataTick()
```

服务器根据协议版本做 `AXES_OFFSET`、`SENSOR_OFFSET_CORRECTION` 等坐标修正，然后将数据放入 `Tracker`。

## 3. Tracker 校准、重置和滤波

主要文件：

```text
server/core/src/main/java/dev/slimevr/tracking/trackers/Tracker.kt
server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt
```

`Tracker.getAdjustedRotation()` 产生用于人体骨骼的有效旋转，概念顺序为：

```text
原始 Rotation
→ Stay Aligned yaw 修正
→ mounting orientation
→ gyroFix / attachmentFix
→ yawFix、全量重置
→ 漂移补偿
→ filteringHandler 滤波
```

最终会调用 `resetsHandler.getReferenceAdjustedDriftRotationFrom(rot)`。`getRawRotation()` 仍代表未经 Server 修正的 Tracker 旋转。

## 4. HumanSkeleton：Tracker 到人体骨骼

主循环位于 `server/core/src/main/java/dev/slimevr/VRServer.kt`：

```kotlin
tracker.tick(...)
humanPoseManager.update()
vMCHandler.update()
```

解算相关文件：

```text
server/core/src/main/java/dev/slimevr/tracking/processor/HumanPoseManager.kt
server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/HumanSkeleton.kt
```

`HumanSkeleton.updatePose()` 执行 `updateTransforms()`、`updateBones()`、`updateComputedTrackers()`、`legTweaks.tweakLegs()` 和 `localizer.update()`。它根据 Tracker role、人体比例和骨骼约束计算头颈、脊柱、髋部、四肢和脚部；缺少 Tracker 时会使用替代 Tracker 或插值。腿部还会经过地面、防滑、脚掌固定、Toe Snap 等后处理。

所以“Tracker 姿态 → 人体骨骼”的核心代码在 SlimeVR Server 的 `HumanSkeleton` 及其 Bone、LegTweaks、配置代码中。

## 5. VMC：SlimeVR 输出到 Unity

文件：`server/core/src/main/java/dev/slimevr/osc/VMCHandler.kt`。

`VMCHandler.update()` 通过 OSC 发送：

```text
/VMC/Ext/Root/Pos
/VMC/Ext/Bone/Pos
/VMC/Ext/Tra/Pos
/VMC/Ext/Hmd/Pos
/VMC/Ext/Con/Pos
```

骨骼包包含名称、位置和四元数。发送前会做 VMC 坐标变换，例如位置 `z → -z`，旋转的 `z`、`w` 分量也按 VMC 约定取反。代码中的发送间隔判断约为 3 ms，即约 200 Hz。

## 6. Unity：接收 VMC 并驱动 Avatar

相关文件：

```text
Assets/EVMC4U/ExternalReceiver.cs
Assets/EVMC4U/DeviceReceiver.cs
Assets/slimeVR/Scripts/VmcOscDebugBridge.cs
Assets/slimeVR/Scripts/VmcTrackerPoseSource.cs
```

`ExternalReceiver` 处理 `/VMC/Ext/Bone/Pos`、`/VMC/Ext/Root/Pos`，将消息写入对应 Transform 的 `localPosition` 和 `localRotation`，并可使用 `BoneFilter` 滤波。此时 Unity Avatar 已经呈现 SlimeVR 计算的人体动作。

## 7. Unity 并行录制三条数据流

主控制器是 `Assets/slimeVR/Scripts/RecordingController.cs`。点击开始录制后调用 `poseRecorder.StartRecording()`；录制期间按采样率调用：

```csharp
CaptureTrackerFrame();
CaptureRawImuFrame();
```

### 7.1 Avatar 姿态：pose.json

`Assets/slimeVR/Scripts/PoseFbxRecorder.cs` 的 `Sample()` 记录所有 Transform 相对路径、局部位置/旋转/缩放、根节点姿态和 Humanoid muscles。运行时保存为 `RuntimePoseRecordingPackage`：

```text
%LOCALAPPDATA%/SpineFlow/PoseRecordings/*.pose.json
```

这是后续 FBX 和 Motion 导出的主要输入。

### 7.2 Tracker 轨迹：tracker.json

`Assets/slimeVR/Scripts/TrackerRecordingData.cs` 定义 `TrackerRecordingPackage`。每帧记录 Tracker 名称、Unity/VMC 坐标系位置和旋转，保存到：

```text
%LOCALAPPDATA%/SpineFlow/TrackerRecordings/*.tracker.json
```

它用于 Tracker 轨迹预览和分析，不是 Avatar 动作文件。

### 7.3 Raw IMU：raw-imu.json

相关文件：

```text
Assets/slimeVR/Scripts/RawImuDataSource.cs
Assets/slimeVR/Scripts/SlimeVrRawImuDataSource.cs
Assets/slimeVR/Scripts/WitRawImuDataSource.cs
Assets/slimeVR/Scripts/RawImuRecordingData.cs
```

优先通过 SolarXR WebSocket `ws://127.0.0.1:21110` 读取 SlimeVR 的直接 Tracker 数据，包含原始/线性加速度、原始角速度、原始磁场、Rotation、RotationReferenceAdjusted、RotationIdentityAdjusted、采样时钟、电量和设备信息。WitMotion 是旧硬件兼容回退路径。最终保存为 `RawImuFrame` / `RawImuSensorSample`：

```text
%LOCALAPPDATA%/SpineFlow/RawImuRecordings/*.raw-imu.json
```

当前 Raw IMU 格式版本为 3。

## 8. 停止录制和导出动作包

`RecordingController.StopRecording()` 的关键顺序：

```csharp
poseRecorder.StopRecording();
TrySaveTrackerRecording(...);
TrySaveRawImuRecording(...);
SaveTrajectoryMeta(...);
```

轨迹元数据位于 `Application.persistentDataPath/TrajectoriesMeta`。

运行时导出入口是 `RecordingController.OnExportLatestMotionPackage()`，实际实现位于 `Assets/slimeVR/Scripts/RuntimeMotionPackageExporter.cs`：

1. 读取 `pose.json`；
2. 读取或创建 `keyframes.json`；
3. 校验关键帧、动作 ID 和分段；
4. 根据 Pose 生成 FBX；
5. 生成 `motion.json`；
6. 拷贝 `tracker.json` 和 `raw-imu.json`（若存在）；
7. 写入 `manifest.json`；
8. 计算 SHA256 并校验动作包。

动作包契约在 `Assets/slimeVR/Scripts/MotionPackageContract.cs`。典型结构：

```text
Recordings/actionId_displayName/v1/
├── manifest.json
├── actionId_v1.fbx
├── actionId_v1.motion.json
├── actionId_v1.tracker.json
├── actionId_v1.raw-imu.json
└── actionId_v1.keyframes.json
```

## 9. 文件职责

| 文件 | 内容 | 用途 |
|---|---|---|
| `*.pose.json` | Unity Avatar 每帧骨骼姿态 | 回放、FBX 和 Motion 导出输入 |
| `*.tracker.json` | Unity 接收到的 Tracker 位姿 | Tracker 轨迹预览 |
| `*.raw-imu.json` | 传感器数据和融合四元数 | 回放、诊断、重建 |
| `*.motion.json` | 标准化动作姿态 | 管理端/康复端播放 |
| `*.fbx` | 3D 动画 | 动画交换 |
| `*.keyframes.json` | 关键帧与分段设置 | 动作编辑 |
| `manifest.json` | 版本、文件引用、哈希 | 导入校验 |

## 10. 故障定位顺序

1. 没有 Tracker 数据：检查固件、UDP 连接和 `TrackersUDPServer.processPacket()`。
2. Tracker 有旋转但人物错误：检查 `TrackerResetsHandler`、佩戴方向、Yaw reset、人体比例和 Tracker role。
3. SlimeVR 正确但 Unity 错误：检查 `VMCHandler` 坐标转换、`ExternalReceiver` 映射和 `BoneFilter`。
4. Unity 画面正确但导出错误：检查 `PoseFbxRecorder.Sample()`、采样率、Avatar Animator 和 `pose.json`。
5. 动作包缺少 Raw IMU：检查 SolarXR `ws://127.0.0.1:21110` 连接和传感器数量配置。

## 边界

Unity 侧负责接收 SlimeVR 输出、驱动 Avatar、并行记录 Tracker/Raw IMU/Avatar 姿态以及导出动作包。人体骨骼解算主要在 SlimeVR Server；IMU 原始采样到融合四元数通常在 Tracker 固件中完成。
