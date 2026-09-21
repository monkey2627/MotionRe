# 手机端 MobilePoser → SMPL 验证应用

这是独立的安卓应用：模拟 IMU 输入、ONNX 推理、SMPL 参数转换和文件导出都在手机内完成。运行时无需 Python、电脑、网络、SlimeVR 或 Unity。本次不含人体渲染，也没有用真实 IMU 验证动作精度。

## 安装和使用

构建产物：`app/build/outputs/apk/debug/app-debug.apk`。安装后：

1. 选择佩戴布局，默认 `wrists_thighs_waist`。
2. 点击“开始推理”。内置每套布局 180 帧、30 Hz 的 surface 合成 IMU 片段。
3. 界面显示 SMPL 参数、实际处理帧率、推理耗时与内存；结束后点击“导出最近 SMPL JSONL”，通过安卓文件选择器保存。
4. 取消“按 30 Hz 回放”可逐帧尽快推理。选中“连续运行 10 分钟”会重复片段，并可导出性能报告。

每次开始、切换布局或循环片段都会重置窗口。运行期间禁用布局切换。点击停止或离开应用即停止；演示版不做后台采集。历史输出保存在应用私有目录，卸载应用会删除，请先导出需要的文件。

CPU 基线：Android 8.0 / API 26 及以上、ARM64；同时打包 x86_64 供模拟器验证。32 位 ARM 不在本次支持范围。每个姿态模型为 5,395,112 参数、约 21.6 MB，6 套全部打包但只加载所选的一套。应用并不保证所有设备实时 30 Hz；落后时降低回放速度，后台没有无限增长的输入队列。

## 输出契约

每行一个 JSON 对象：

| 字段 | 定义 |
|---|---|
| `model_type` | `smpl`，不是 SMPL-X |
| `global_orient` | 3 维根关节轴角，弧度 |
| `body_pose` | 69 维，SMPL 关节 1–23 的局部轴角 |
| `transl` | `[0,0,0]`，没有预测全局位移 |
| `betas` | 10 个零，没有预测体型 |
| `smpl_local_rotations` | 24×3×3 局部旋转矩阵，按行展平为 216 维，供数值核对 |
| `identity_joint_indices` | `[7,8,10,11,20,21,22,23]`，沿用原模型的单位旋转规则 |
| `input_time_seconds` | 当前输入帧的时间 |
| `pose_time_seconds` | 输出姿态对应的输入时间，不是推理完成时间 |
| `inference_ms` | 窗口更新、张量创建及 ONNX 执行耗时，不含 JSON 写出 |
| `layout`、`frame_index` | 布局和从零开始的输出序号 |

坐标系沿用训练数据的 Y-up 世界坐标；关节旋转相对父节点，不转换到 Unity 坐标。完整姿态可以拼接为 `global_orient + body_pose` 的 72 维向量。给 SMPL 模型调用时为这些数组补上 batch 维度即可；应用本身不生成 SMPL 网格。

在线窗口为 `[1,45,60]`，取索引 40。稳定运行后输出落后当前输入 4 帧（约 133 ms），再加实际计算耗时。启动时用首帧重复填满窗口，前四帧输出时间均指向首帧。保留双向 LSTM，不采用会改变 checkpoint 行为的单帧隐状态复用。

## 自定义模拟输入

使用 `assets_generated/<layout>.input.json` 作为完整模板，点击“导入 IMU JSON”。文件格式：

```json
{
  "format_version": 1,
  "layout": "wrists_thighs_waist",
  "fps": 30,
  "sensor_labels": ["lw", "rw", "lt", "rt", "waist"],
  "acceleration_units": "m/s2",
  "acceleration_space": "world_linear",
  "orientation_space": "bone_calibrated",
  "frames": [
    {
      "time_seconds": 0.0,
      "acceleration": [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
      "orientation": [
        [[1,0,0],[0,1,0],[0,0,1]],
        [[1,0,0],[0,1,0],[0,0,1]],
        [[1,0,0],[0,1,0],[0,0,1]],
        [[1,0,0],[0,1,0],[0,0,1]],
        [[1,0,0],[0,1,0],[0,0,1]]
      ]
    }
  ]
}
```

五个传感器必须按照布局顺序排列。`wrists` 对应训练中的前臂佩戴位置；`waist` 为 spine1 对应腰部信号，不是头部。支持的顺序见 `ImuClip.layouts` 和导出的 `manifest.json`。

加速度必须已去除重力、转换到世界坐标，单位 m/s²；不要提前除以 30。旋转必须是已经做过骨骼佩戴校准的旋转矩阵，不能直接填未校准四元数、欧拉角或 SlimeVR 原始包。应用只负责归一化与拼接，不假装已经实现真实传感器校准。输入限定 20 MiB、最多 18,000 帧，时间间隔为 1/30 秒。

## 从源码构建

在仓库根目录，使用已安装原项目依赖的 Python 环境：

```bash
conda activate mobileposer
python -m pip install -r android/requirements-export.txt
python -m mobileposer.mobile_export
```

默认从 `checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth` 提取关节与姿态网络，从 `data/no_head_5imu_surface_dip_test` 生成模拟输入。导出时独立调用原项目的 packed-sequence 姿态分支作为参考，逐帧核对全部 6 套模型，误差超限则失败。

产物位于 `android/assets_generated/`：6 个 ONNX、对应输入和参考输出、带模型/权重 SHA256 的 manifest、桌面验证报告。生成文件不纳入 Git，完整构建前必须生成全部 6 套。

安装 JDK 17 或 21、Android SDK platform 35 和 build-tools，配置 `ANDROID_HOME` 或 `android/local.properties` 的 `sdk.dir`，然后：

```bash
cd android
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Gradle 8.9、AGP 8.7.3、Kotlin 2.0.21、ONNX Runtime 1.19.2 已固定。Python 环境为 3.9，因此桌面和安卓统一选用支持它的 ORT 1.19.2。运行时不依赖 CUDA。

## 安卓数值验收

连接设备或启动模拟器：

```bash
cd android
./gradlew :app:connectedDebugAndroidTest
mkdir -p captures
adb exec-out run-as org.mobileposer.smpl cat files/validation_report.android.json > captures/validation_report.android.json
adb exec-out run-as org.mobileposer.smpl cat files/wrists_thighs_waist.android.jsonl > captures/wrists_thighs_waist.android.jsonl
cd ..
python -m mobileposer.verify_android_output \
  android/captures/wrists_thighs_waist.android.jsonl \
  android/assets_generated/wrists_thighs_waist.reference.json
```

设备测试覆盖 6 套布局、每套 180 帧、模型切换、重置和应用启动。桌面复核脚本还会将输出轴角转回旋转矩阵，与原 PyTorch 参考比较，避免只验证中间矩阵而漏掉 SMPL 参数转换错误。

没有硬件加速的模拟器可以用 `./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.sparseFrames=true` 做抽样验证：每套只推理索引 0、4、44、45、90、179，并额外验证重置；其余帧仍更新窗口。复核抽样 JSONL 时给 Python 脚本加 `--allow-sparse`。这不等同于安卓完整 180 帧测试。

性能报告包含推理 P50/P95、实际帧率、每秒帧率和内存 PSS，便于观察持续运行退化。模拟器的速度不能代表 ARM 手机；真实 IMU 的精度、真机发热降频和各档设备兼容性需要额外实测。具体完成情况见 `VALIDATION.md`。
