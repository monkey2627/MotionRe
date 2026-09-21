# 全部 ours 动作联合微调

入口：`python -m mobileposer.finetune_ours_multi`。只需运行此入口，自动完成 IMU 适配、Unity HumanPose 导出、SMPL 伪标签生成、训练、所有视频生成。

**每种 IMU layout 只训练一个联合模型，所有符合该布局的 ours 动作共同更新这一个模型。不是每个动作训练一个模型。** 六种布局的槽位不同，仍分别训练对应的六个模型。

在项目根目录运行，使用已有 mobileposer 环境：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python \
-m mobileposer.finetune_ours_multi \
--raw-root data/raw/ours \
--output results/ours_multi_finetune \
--epochs 120 --lr 0.0003 --device cuda:0
```

默认递归搜索 manifest.json，因此 `ours/4_臀桥/1` 和 `ours/1` 都会纳入。每个 epoch 随机遍历全部可用动作，每段动作更新一次；独立处理整段序列，不跨动作拼接、不截短最终视频。较长动作的整段反向传播需要更多显存。

默认从 `checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth` 初始化，有利于检查多动作联合适配，而不是延续之前单动作过拟合偏向。若确实要从上次单动作微调继续：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python \
-m mobileposer.finetune_ours_multi \
--init-root results/ours1_overfit_smoke \
--layouts wrists_thighs_waist wrists_shanks_waist upperarms_thighs_waist wrists_upperarms_waist legs_waist \
--output results/ours_multi_continue --epochs 120 --device cuda:0
```

上次没有脚部微调权重，所以该命令只选五组。默认命令会尝试所有六组。脚部模型只使用确有左右脚实体 IMU 且检查通过的采集。

若要延续本脚本产生的联合模型，使用 `--init-root results/ours_multi_finetune/layouts`，并通过 `--layouts` 指定上一轮实际完成的布局。加载模型权重，优化器重新初始化。

## 输出

- `README.md`：所有动作、布局的指标与视频链接。
- `catalog.json`：发现的全部采集包、可用布局、跳过原因。
- `captures/<采集ID>/`：输入、参考导出、targets.pt、伪标签检查视频。
- `layouts/<layout>/model_finetuned.pth`：该布局的联合微调模型。
- `layouts/<layout>/history.json`：训练记录。
- `layouts/<layout>/<采集ID>/before.mp4`：微调前。
- `layouts/<layout>/<采集ID>/after.mp4`：微调后。
- `layouts/<layout>/<采集ID>/before_after.mp4`：左侧微调前、右侧微调后；各自蓝色 HumanPose、红色预测。
- 同目录 `before.pt` / `after.pt`：完整预测参数。
- `comparison.csv` / `summary.json`：全部已完成结果。

所有可用动作都会输出全时长视频，15 FPS 显示、30 FPS 推理；没有仅输出前几段的限制。输出目录必须为空，避免覆盖已有实验。中断后可查看已完成布局的结果；目前不支持自动断点续训，重新运行应指定新目录。

## 训练与检查约定

沿用之前 smoke test 的 joints+poser 联合损失：关节位置、骨段方向6D、SMPL FK 位置；冻结 velocity/contact，保留原始 checkpoint，监督来自 motion.json 的 HumanPose→SMPL 伪标签。对每个动作重置推理状态，以全部训练动作按帧数加权的伪标签 MPJPE 选择最佳模型。

多动作增加训练覆盖，但这些视频仍是训练集效果，不是独立泛化评估。若要评估泛化，需要额外未参与训练的采集。

不填充缺失/离线传感器，不把手部当作前臂，不将虚拟 tracker 当作实体 IMU。航向拟合不可靠的传感器会导致对应布局被跳过。请检查 catalog.json；不能仅凭有视频就认为所有布局/采集均已使用。

Unity 导出需要本机 Unity Editor 2022.3.13f1 和项目可运行；默认路径与之前验证一致，可用 `--unity-editor` 替换。运行时应关闭占用该 Unity 项目的编辑器实例。

只查看数据发现结果（不导出、不训练）：

```bash
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m mobileposer.finetune_ours_multi --list-only
```

仅代码检查已经完成；本次没有启动 Unity 导出、伪标签拟合或微调。
