# Deep Inertial Poser：从稀疏惯性测量中实时重建人体姿态

> 原文：Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time
> 发表于 SIGGRAPH Asia 2018

## 代码结构

本仓库包含论文配套代码，组织如下：
- [`train_and_eval/`](train_and_eval)：训练与评估神经网络的代码
- [`live_demo/`](live_demo)：实时推理的 Unity 和 Python 脚本
- [`data_synthesis/`](data_synthesis)：从 SMPL 动作序列生成合成 IMU 数据的脚本

## 数据

前往[项目主页](http://dip.is.tuebingen.mpg.de)下载数据集，同时可下载 TotalCapture 数据集的 SMPL 参考参数。TotalCapture 数据预处理见 [`data_synthesis/read_TC_data.py`](data_synthesis/read_TC_data.py)。

## 可视化

数据可用 [aitviewer](https://github.com/eth-ait/aitviewer) 可视化，其示例包含两个 DIP 相关脚本：
- [加载 DIP-IMU 数据集的 SMPL 姿态与 IMU 数据](https://github.com/eth-ait/aitviewer/blob/main/examples/load_DIP_IMU.py)
- [加载 TotalCapture 数据集的 SMPL 姿态与 IMU 数据](https://github.com/eth-ait/aitviewer/blob/main/examples/load_DIP_TC.py)

![DIP-IMU rendering with aitviewer](DIP_IMU_example.gif)

## 联系方式

如有问题请提 issue 或联系：
- [manuel.kaufmann@inf.ethz.ch](mailto:manuel.kaufmann@inf.ethz.ch)
- [yinghao.huang@tuebingen.mpg.de](mailto:yinghao.huang@tuebingen.mpg.de)

## 引用

```bibtex
@article{DIP:SIGGRAPHAsia:2018,
    title = {Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time},
    author = {Huang, Yinghao and Kaufmann, Manuel and Aksan, Emre and Black, Michael J. and Hilliges, Otmar and Pons-Moll, Gerard},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    volume = {37},
    pages = {185:1-185:15},
    publisher = {ACM},
    month = nov,
    year = {2018},
    note = {First two authors contributed equally},
    month_numeric = {11}
}
```
