# IMUTrack-for-Spine（动作录制）


## 项目定位

本项目是「VR 脊柱康复系统」的**动作录制工具**（Unity 端）。管理员佩戴 IMU / SlimeVR 追踪器做康复动作，录制人体运动轨迹，并导出为动作数据包（FBX + 动作 JSON + 追踪器 JSON + 原始 IMU JSON），供管理员端与康复端使用。

## 技术栈

- Unity（含 Editor 扩展菜单）
- WitMotion IMU SDK（`witmotionUnity`，UDP 接收加速度/角速度/磁场/欧拉角）
- SlimeVR VMC 追踪器（`slimeVR`，OSC 姿态流）
- Raw IMU 数据源（`RawImuDataSource`：优先 SlimeVR / SolarXR 直接追踪器数据，WitMotion 作为老硬件兼容回退）
- EVMC4U（外部动作捕捉接收）
- VRM / VRM10 / UniGLTF / VRMShaders（虚拟人模型导入导出）
- RootMotion FinalIK（IK 驱动）
- uOSC（OSC 通信）

## 核心目录

```
Assets/
├── Scenes/SampleScene.unity     # 主场景（录制面板 + 回放）
├── Prefabs/                      # RecordingPanel、TrajectoryListPanel 等 UI
├── slimeVR/Scripts/              # 录制控制与动作包契约（核心）
│   ├── RecordingController.cs    # 录制主控制器（校准/录制/回放/导出）
│   ├── PoseFbxRecorder.cs        # 姿态 FBX 录制
│   ├── MotionPackageContract.cs  # 动作包磁盘契约（与 SpineFlowAdmin 共享）
│   └── Editor/                   # 后处理、导出窗口
├── witmotionUnity/               # WitMotion IMU SDK 与 LegIK 驱动
├── EVMC4U/                       # VMC 外部接收
├── opensim/                      # OpenSim 骨骼几何
├── Models/ / vrmAssets/          # VRM 模型
└── Recordings/                   # 录制产物（.fbx / .motion.json，不纳管）
```

## 运行方式

1. 用 Unity 打开本项目根目录。
2. 打开场景 `Assets/Scenes/SampleScene.unity`。
3. 运行后按录制面板操作：校准 → 开始录制 → 停止录制 → 回放 / 导出。

## 核心流程

1. **校准**：连接 WitMotion IMU 与 SlimeVR，点击校准（`RecordingController.OnCalibrate`）。
2. **录制**：填写教练身高/体重/性别，点击「开始录制」，做动作后点击「停止录制」。
3. **产物**：录制生成
   - `.anim` / `.fbx`（姿态动画）
   - `.motion.json`（动作数据包）
   - `.tracker.json`（追踪器轨迹）
   - 原始 IMU JSON（加速度/角速度/磁场/欧拉角/电量）
4. **动作编辑**：录制后可在后处理窗口标记关键帧 / 分段（`RecordedMotionEditDefinition`，含帧号、保持时长、身体部位、分段时长）。
5. **导出动作包**：Editor 菜单
   - `IMUTrack/Process Latest Recording`（后处理窗口）
   - `IMUTrack/Export Latest Recording Directly`（直接导出最新录制为动作包）

## 动作包（Motion Package）

动作包契约定义在 `Assets/slimeVR/Scripts/MotionPackageContract.cs`（命名空间 `SpineFlow.MotionPackages`），**与 SpineFlowAdmin 共享同一份契约**。

- 存储位置：构建后位于可执行文件同级的
  `Recordings\<actionId>_<displayName>\<version>\`；编辑器内对应项目根目录。
  `Recordings` 不存在时会自动创建，同一动作重复导出时通过版本子目录避免覆盖。
- 每个包包含：`manifest.json`（清单 + SHA256 校验）、`.fbx`、`.motion.json`、可选的 `.tracker.json`
- `actionId` 只能包含小写字母、数字、下划线
- manifest 记录 `actionId`、`displayName`、`version`、`segments`（分段/关键帧区间）

### 导入并回放动作包

1. 在录制界面点击“导入 JSON”。
2. **优先选择动作包版本目录中的 `manifest.json`**。程序会读取其中的
   `motionJsonFile`，校验 SHA256，并导入对应的 `*.motion.json` 姿态帧。
3. 导入完成后，点击新出现的“导入”轨迹右侧播放按钮，打开关键帧编辑界面；
   人物动作的播放、暂停、逐帧定位和关键帧编辑均在该界面完成。

只有单独的动作文件时，也可以直接选择 `*.motion.json`。`*.tracker.json` 只用于
Tracker 物体轨迹预览，`*.raw-imu.json` 只用于原始传感器数据预览，二者都不是
Humanoid 人物动作回放文件。

## 与其它模块的关系

| 模块 | 关系 |
|------|------|
| `SpineFlowAdmin`（PC 管理员端） | 读取/导入本工具导出的动作包，编辑关键帧后发布 |
| `SpineFlowMobile-Admin`（旧手机管理员端） | 旧版参考实现 |

## 已知说明

- `Assets/Recordings/` 下的录制产物（`.fbx`、`.motion.json`）属于**产物，不纳入版本控制**。
- `.anim` 资源回放及部分 FBX 导出流程仅在 **Unity Editor** 下可用；动作包中的
  `*.motion.json` 会转换为运行时 Humanoid 姿态，可在 Editor 和支持的 Standalone 中回放。
- 录制依赖 WitMotion IMU 与 SlimeVR 设备，未连接设备时仅能回放已有轨迹。

## 注意事项

- 本项目仓库启用了 Git LFS，`.vrm` / `.vtp` / `.obj` / `.fbx` / `.png` 等大文件走 LFS 存储。
