# DIP：数据合成

## 功能说明

从 SMPL 格式的 MoCap 序列生成合成 IMU 数据，与 DIP 训练时使用的流程相同。

**核心逻辑**（`genSynData.py`）：
1. 加载 SMPL 模型，对每帧姿态做正向运动学，得到所有关节的全局变换矩阵
2. 提取 6 个传感器位置（左腕/右腕/左腿/右腿/头/骨盆）的旋转矩阵作为 **orientation**
3. 通过二阶差分计算对应顶点的线加速度作为 **acceleration**：
   ```
   acc[t] = (v[t+1] + v[t-1] - 2*v[t]) / (Δt²)
   ```
4. 将动作帧率统一插值到 60 FPS

---

## 运行步骤

### 第一步：下载 SMPL 模型

去 [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) 下载 `SMPL_python_v.1.0.0`，解压后放到本目录下：

```
data_synthesis/
└── SMPL_python_v.1.0.0/
    └── smpl/
        ├── models/
        │   ├── basicModel_m_lbs_10_207_0_v1.0.0.pkl
        │   └── basicModel_f_lbs_10_207_0_v1.0.0.pkl
        └── smpl_webuser/
```

### 第二步：准备 MoCap 数据

仓库自带示例文件 `Jog_1.pkl`（AMASS HumanEva 子集），可直接使用。

如需使用自己的 AMASS 数据，从 [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) 下载后，每个 `.npz` 需转换为包含以下字段的 `.pkl`：

```python
{
  'gender': 'male' 或 'female',
  'betas': 体型参数, shape=(10,),
  'poses': 姿态参数列表，每帧 shape=(72,)，轴角格式,
  'frame_rate': 原始帧率（如 120）
}
```

### 第三步：安装依赖

```bash
pip install chumpy opencv-python numpy
```

> 注意：`chumpy` 是旧版 SMPL 的依赖，Python 3.8 下可能需要手动修复兼容问题。

### 第四步：运行

```bash
cd DIP-IMU/data_synthesis/
python genSynData.py ./Jog_1.pkl ./Jog_1_synthesis.pkl
```

**输出文件** `Jog_1_synthesis.pkl` 包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `ori` | `[T, 6, 3, 3]` | 6 个 IMU 的旋转矩阵（每帧） |
| `acc` | `[T, 6, 3]` | 6 个 IMU 的线加速度 |
| `poses` | `[T, 135]` | 15 个关节 × 9D 旋转矩阵 |

---

## 预处理 TotalCapture

使用 `read_TC_data.py` 预处理 TotalCapture 数据集。

---

## 与 MotionRecover 的关系

`template_pose_to_imu()` 函数（Week 3 实现目标）直接复用本文件的逻辑：给定模板姿态 `T_t`，调用 `compute_imu_data()` 推算出理论 IMU 信号，填入 MobilePoser 的零掩码位置。核心参考函数为 `genSynData.py` 第 38–70 行的 `get_ori_accel()`。
