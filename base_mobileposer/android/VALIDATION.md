# 验证记录

验证日期：2026-09-20。桌面环境：Linux x86_64、Python 3.9、PyTorch 2.1.2、ONNX Runtime 1.19.2。安卓构建：Gradle 8.9、AGP 8.7.3、Kotlin 2.0.21、compile/target SDK 35、min SDK 26。

## 模型数值对齐

每套布局测试连续 180 帧，共 1,080 帧。参考来自原仓库的关节网络、姿态网络和 SMPL 全局转局部旋转函数；输入窗口与 `forward_online` 相同。

| 布局 | ONNX 与原 PyTorch 的最大矩阵元素误差 |
|---|---:|
| wrists_thighs_waist | 4.7684e-7 |
| wrists_shanks_waist | 3.8743e-7 |
| wrists_feet_waist | 5.3644e-7 |
| upperarms_thighs_waist | 4.4703e-7 |
| wrists_upperarms_waist | 4.7684e-7 |
| legs_waist | 4.9174e-7 |

全部低于 1e-4 验收阈值。每个 ONNX 为 21,626,409 字节、5,395,112 个参数，输入 `[1,45,60]`，输出 `[24,3,3]`。原 checkpoint 未修改。

详细数据和权重/ONNX SHA256 见 `assets_generated/validation_report.json` 和 `assets_generated/manifest.json`。

## 安卓构建与测试

- `:app:assembleDebug`：成功，生成包含 ARM64 和 x86_64 的 APK。
- `:app:assembleDebugAndroidTest`：成功。
- `:app:testDebugUnitTest`：5 项测试通过，覆盖归一化与拼接、输入拒绝、窗口/重置、轴角奇异点、SMPL 输出字段。
- `:app:lintDebug`：通过，无错误；有依赖更新、国际化、图标和备份配置提示。
- Python 输出复核工具的 3 项测试通过：正确轴角/矩阵一致性、错误轴角拒绝、空输出拒绝。
- `apksigner verify`：APK 签名校验通过，使用 Android Debug 测试签名。

## 安卓模拟器执行

Android 11 / API 30 x86_64 软件模拟器中，APK 和测试包均安装成功。已取得默认布局 `wrists_thighs_waist` 的 6 个实际推理输出：输入索引 0、4、44、45、90、179。未推理的中间帧仍按顺序更新 45 帧窗口。

独立复核脚本对比了输出矩阵和由最终 SMPL 轴角重建的矩阵，最大元素误差 **3.5585e-7**，低于 1e-4。这证明默认模型的安卓加载、模拟输入、ONNX 推理和 SMPL 参数输出链路可以执行。

实际输出：`captures/wrists_thighs_waist.sampled.android.jsonl`；验证摘要：`captures/validation_report.android.sampled.json`。模拟器测试自身保存的首套布局结果为 `captures/validation_report.android.progress.json`，该布局的重置检查也已通过。

主 Activity 启动成功，系统报告其为前台 resumed Activity；模拟器出现 System UI ANR，因此不将截图作为完整界面验收。当前账户无 `/dev/kvm` 权限，软件模拟的单次推理耗时达数十秒，不能作为手机性能指标。完整 6 套布局的安卓 instrumentation suite 未跑完；其余 5 套的本次数值验证范围仅为桌面。完整/抽样测试命令均已提供，供加速模拟器或实体设备继续验收。

## 尚未证明的内容

ARM64 真机推理速度、连续 10 分钟温升/降频、不同档位手机兼容性、真实 IMU 动作精度尚未验证。APK 编译成功和桌面 ONNX 结果不能替代这些测试。应用提供 10 分钟循环回放和性能报告导出功能，供真机验收。
