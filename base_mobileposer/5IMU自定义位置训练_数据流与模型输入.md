# 5-IMU 自定义位置训练：数据流与模型输入

基于 `base_mobileposer/` 现有代码梳理「自定义 6 组表面 IMU 布局」训练链路：raw 数据 → AMASS/DIP 合成 IMU → 训练集 `.pt` → `PoseDataset`/`DataLoader` → 模型输入。路径均相对 `base_mobileposer/`。

## 0. 两套并行的预处理体系

| | 标准 5 感测点（原版 MobilePoser） | 自定义 6 组表面布局（本项目） |
|---|---|---|
| 脚本 | `mobileposer/process.py` | `mobileposer/no_head_layouts.py` + `mobileposer/surface_imu.py` |
| 传感器位置 | 固定：左/右手腕、左/右大腿、头、骨盆（6 点，训练只用前 5 个） | `LAYOUTS` 定义的 5 个自定义位置 + 骨盆（同样 6 slot，训练只用前 5 个） |
| 输出目录 | `data/processed_datasets/*.pt`（每个 AMASS 子集一个文件） | `data/no_head_5imu[_surface]_processed/<layout_name>/*.pt` |
| 驱动脚本 | `train.py`（直接读 `paths.processed_datasets`） | `run_no_head_5imu_experiments.py`（环境变量重定向 `paths.processed_datasets`） |

你训练的 6 组数据就是 `no_head_layouts.LAYOUTS` 里的 6 个布局，走右边这条链路，两套体系公用同一个 `data.py`/训练代码。

## 1. Raw 数据

- AMASS：`data/raw/AMASS/<dataset_name>/*/*_poses.npz`（`poses`, `trans`, `betas`, `mocap_framerate`）。子集名列表见 `config.py:datasets.amass_datasets`。
- DIP-IMU：`data/raw/DIP_IMU/<subject>/<motion>.pkl`（`imu_acc`, `imu_ori`, `gt` 轴角姿态）。DIP 只在原始 6 个位置有真实传感器，自定义布局位置没有真实信号，只能用 DIP 的 GT SMPL 姿态重新合成。

## 2. 自定义布局定义（`no_head_layouts.py:24-49`）

```python
LAYOUTS = {
    "wrists_thighs_waist":    {"labels": ["lw","rw","lt","rt","waist"], "joints": [18,19,1,2,3]},
    "wrists_shanks_waist":    {"labels": ["lw","rw","ls","rs","waist"], "joints": [18,19,4,5,3]},
    "wrists_feet_waist":      {"labels": ["lw","rw","lf","rf","waist"], "joints": [18,19,7,8,3]},
    "upperarms_thighs_waist": {"labels": ["lu","ru","lt","rt","waist"], "joints": [16,17,1,2,3]},
    "wrists_upperarms_waist": {"labels": ["lw","rw","lu","ru","waist"], "joints": [18,19,16,17,3]},
    "legs_waist":             {"labels": ["lt","rt","ls","rs","waist"], "joints": [1,2,4,5,3]},
}
```
`joints` 是 SMPL 骨骼 id（0 骨盆，1/2 左右髋，3 spine1，4/5 左右膝，7/8 左右踝，16/17 左右肩，18/19 左右肘）。每个布局固定 5 个传感器位置，按 `labels` 顺序对应模型输入的前 5 个 slot。

## 3. IMU 合成：`joint` 模式 vs `surface` 模式

`no_head_layouts.py:_synthesize()` 按 `--imu-mode` 二选一，两者共用同一个五点差分公式 `_syn_acc()`（`smooth_n=4` 平滑差分，与 `process.py:_syn_acc()` 一致）：

- **`joint`（关节基线）**：用 `body_model.forward_kinematics` 取骨骼全局旋转 `r[:, ids]` 和关节位置 `joint[:, ids]`（`ids = layout['joints'] + [0]`，末尾附加骨盆 id 0），对关节位置做差分得到加速度。
- **`surface`（皮肤表面）**：`surface_imu.py:surface_kinematics()` 用冻结的网格贴附点（`configs/no_head_surface_imu.json`，barycentric weights 固定在某个三角面上，SHA256 校验对应训练用的 SMPL 模型）取该点的世界坐标，对该表面点做差分得到加速度，比关节中心点更接近真实贴皮肤的效果（含肢体转动带来的表面切向加速度）。**该模式下朝向（ori）在数值上与 `joint` 模式相同**（`surface_kinematics` 里的安装标定旋转 `mount_rotation` 是正交矩阵，乘完再乘其转置会正好抵消），即两种模式只有 `acc` 不同，`ori` 都是骨骼全局旋转本身。

`configs/no_head_surface_imu.json` 由 `python -m mobileposer.surface_imu --initialize` 一次性生成，之后所有布局共用这份贴附点配置。

## 4. 生成训练集：`generate_layout_dataset()`（AMASS）/ `generate_dip_layout_dataset()`（DIP）

```
generate_layout_dataset(layout_name, output_dir, mode="joint"|"surface", ...)
  → 对每个 AMASS 子集：
      1. 读取 raw npz，降采样到 30fps，对齐 AMASS→DIP 全局坐标系（与 process.py 一致的 amass_rot）
      2. axis-angle → rotation matrix（局部旋转）
      3. body_model.forward_kinematics（joint 模式）或 surface_kinematics（surface 模式）
      4. 五点差分得到 acc
      5. _foot_ground_probs() 算脚接触地面概率（与 process.py 相同阈值 0.008m）
      6. 落盘：{synthesis, layout, layout_labels, layout_joints, joint, pose, shape, tran, acc, ori, contact}
         → data/no_head_5imu[_surface]_processed/<layout_name>/<AMASS子集>.pt
```

`acc`/`ori` 的 shape 是 `[N, 6, 3]` / `[N, 6, 3, 3]`：**前 5 个是该布局的 5 个自定义传感器（按 `labels` 顺序），第 6 个固定是骨盆/root**（对应 `run_no_head_5imu_experiments.py` 注释：`slot order: 0..4 are the five learned inputs; pelvis/root is stored as slot 5 for compatibility`）。这个 6-slot 结构和原版 `process.py` 的 `vi_mask/ji_mask`（lw, rw, lt, rt, head, pelvis）对齐，所以两套数据能复用同一个 `data.py`。

`synthesis` 字段记录 `mode/fps/acceleration/orientation/smpl_sha256`，`validate_cached_dataset()` 用它做缓存校验（`--overwrite` 强制重跑）。

DIP 侧 `generate_dip_layout_dataset()` 逻辑相同，姿态来源换成 DIP 的 `gt` 轴角（DIP 没有 `tran`，恒为 0），只用于该布局的 finetune/eval，不参与主训练。

## 5. `PoseDataset` 如何读取这些 `.pt`（`mobileposer/data.py`）

训练某个布局时，`run_no_head_5imu_experiments.py` 把 `MOBILEPOSER_PROCESSED_DATASETS` 指向 `data/no_head_5imu[_surface]_processed/<layout_name>/`，并设 `MOBILEPOSER_TRAIN_COMBOS=all_5imu`（`config.py:79-80`）：

```python
if os.environ.get("MOBILEPOSER_TRAIN_COMBOS") == "all_5imu":
    combos = {"all": [0, 1, 2, 3, 4]}
```
这一步把原本用于「缺传感器鲁棒性训练」的 12 组子集 mask（`amass.combos`）关掉，只保留「5 个传感器全给」这一种组合，因为自定义布局是固定 5 个物理点位。

`PoseDataset._prepare_dataset()`：`fold='train'` 时 `data_folder` 即上面重定向的目录，直接列出该目录下所有 `.pt` 文件逐个 `torch.load` 读入 `{acc, ori, pose, tran, joint, contact}`。

`_process_file_data()`（`data.py:57-67`）：
```python
acc, ori = acc[:, :5]/amass.acc_scale, ori[:, :5]      # 只取前 5 个 slot，丢弃第 6 个骨盆槽位；acc 除以 30 归一化
pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216))  # 局部旋转矩阵 → 全局
pose = pose_global.view(-1, 24, 3, 3)                   # 训练时姿态目标用全局旋转
```
`joint` 在这里被 `forward_kinematics` 的返回值覆盖，且调用时不传 `shape`/`tran`，所以训练用的关节位置 GT（`joint_outputs`）统一用零体型骨架、原点为中心重新计算，与数据来自哪个 AMASS 子集的真实体型无关；体型信息只体现在 `acc`/`ori` 的合成过程里。这是 `data.py` 本身的行为，标准数据和自定义布局数据一致。

`_process_combo_data()`（`data.py:69-85`）：
```python
for _, c in self.combos:                                # combos={"all":[0,1,2,3,4]} → 只循环 1 次，不做 mask
    combo_acc, combo_ori = zeros_like(acc), zeros_like(ori)
    combo_acc[:, c] = acc[:, c]
    combo_ori[:, c] = ori[:, c]
    imu_input = cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)   # [N,15] ⊕ [N,45] → [N,60]
    data_len = window_length(=125) if 训练 else 整段长度
    按 data_len 切成定长窗口，分别 extend 进 data['imu_inputs'/'pose_outputs'/'joint_outputs'/'tran_outputs']
    if 非 evaluate/finetune:
        root_vel = diff(tran); vel = diff(joint), vel[:,0]=root_vel
        extend data['vel_outputs'] (缩放 fps/vel_scale=30/2=15)
        extend data['foot_outputs'] = contact
```

**`imu_input` 的 60 维是分块拼接 `[acc(15) | ori(45)]`**，不是每个传感器 12 维交叉排列：先把 5 个传感器的加速度按 `labels` 顺序拼成 15 维，再把 5 个传感器的 3×3 旋转矩阵展平拼成 45 维，两段拼在一起才是 60 维。

`__getitem__()` 把姿态转成 6D 表示（`pose(144D=24*6)`），返回 `(imu, pose, joint, tran, vel, contact)`（训练）或 `(imu, pose, joint, tran)`（evaluate/finetune）。

## 6. DataLoader / 批处理

- `PoseDataModule.setup('fit')`：`PoseDataset(fold='train')` 构建全部窗口后 `random_split` 成 90% train / 10% val。
- `pad_seq()` collate：每条窗口长度可能不足 125（原始序列尾部残余），用 `nn.utils.rnn.pad_sequence` 对 batch 内的 `imu/poses/joints/trans/vels/foot_contacts` 做变长 padding，记录真实长度供 RNN 的 `pack_padded_sequence` 用。
- `DataLoader(batch_size=256, num_workers=8, shuffle=True, drop_last=True)`（`train_hypers`）。
- batch 结构：`((inputs, input_lengths), (outputs, output_lengths))`，`inputs.shape=[B,S,60]`（`S`≤125），`outputs['poses'].shape=[B,S,144]`，`outputs['joints'].shape=[B,S,24,3]`。

## 7. 模型输入（4 个独立子模块，`mobileposer/models/*.py`）

底层都是同一个 `RNN`（`Linear→Dropout→ReLU→双向LSTM(2层)→Linear`）：

| 模块 | 输入维度 | 输出维度 | hidden | bidirectional |
|---|---|---|---|---|
| `Joints` | 60（纯 IMU） | 72 (`24*3` 关节位置) | 256 | 是 |
| `Poser` | 132 (`72关节位置⊕60 IMU`) | 96 (`joint_set.n_reduced(16)*6`) | 256 | 是 |
| `FootContact` | 132 | 2 (左右脚接触概率 logit) | 64 | 是 |
| `Velocity` | 132 | 72 (`24*3` 关节速度) | 256 | 否（单向） |

`Joints` 只吃 60D IMU。`Poser`/`FootContact`/`Velocity` 的 132D 输入是 `[72D关节位置 ⊕ 60D IMU]`，但这个"关节位置"来源在训练和推理两个阶段不同：

- **单模块训练**（`train.py` 默认逐个训练 poser/joints/foot_contact/velocity，四者互不依赖）：`Poser`/`FootContact`/`Velocity` 用的是 **dataset 给出的 GT 关节位置**（`joint_outputs`）加噪声（Poser/FootContact std=0.04，Velocity std=0.025）后拼接 IMU，不依赖 `Joints` 模块的输出。
- **联合推理**（`combine_weights.py` 合并权重后的 `MobilePoserNet.forward()`）：先跑 `Joints(imu)` 得到 `pred_joints`，再拼接 IMU 喂给其余三个模块，此时才真正级联。

`Poser` 只回归 `joint_set.reduced` 16 个关节（`[0,1,2,3,4,5,6,9,12,13,14,15,16,17,18,19]`）的 6D 旋转（96D），其余 8 个关节在 `_reduced_global_to_full()` 里补单位阵，再通过 `reduced_pose_to_full` + `inverse_kinematics_R`（全局→局部）还原成完整 24 关节局部姿态。`Joints`/`Velocity` 的监督信号直接来自 dataset 的 `joint_outputs`/`vel_outputs`。

## 8. 训练启动方式

单个布局手动跑（等价于 `run_no_head_5imu_experiments.py` 内部做的事）：
```bash
python -m mobileposer.no_head_layouts --layout wrists_thighs_waist --imu-mode joint   # 生成训练集
MOBILEPOSER_PROCESSED_DATASETS=data/no_head_5imu_processed/wrists_thighs_waist \
MOBILEPOSER_CHECKPOINT_DIR=checkpoints/no_head_5imu/wrists_thighs_waist \
MOBILEPOSER_TRAIN_COMBOS=all_5imu \
python train.py                     # 依次训练 poser/joints/foot_contact/velocity，各 60 epoch
python combine_weights.py --checkpoint-path checkpoints/no_head_5imu/wrists_thighs_waist/<run_id>
```

六组布局一键跑全流程（预处理→训练→合并权重→推理→评估）：
```bash
python -m mobileposer.run_no_head_5imu_experiments --layouts all --imu-mode joint
python -m mobileposer.run_no_head_5imu_experiments --layouts all --imu-mode surface
# 支持 --gpus 0 1 2 把 6 个布局轮询分摊到多卡并行跑
```
该脚本对每个布局重定向 `MOBILEPOSER_PROCESSED_DATASETS/MOBILEPOSER_CHECKPOINT_DIR/MOBILEPOSER_TRAIN_COMBOS` 三个环境变量后 `subprocess` 调用 `train.py`，训练完自动 `combine_weights.py` → `infer_mobileposer.py --combos all --n-sensors 5` → `evaluate_no_head_5imu.py`。

## 9. 端到端数据流总图

```
raw/AMASS *.npz  ──┐                          raw/DIP_IMU *.pkl (仅 GT pose)
                    │                                     │
   no_head_layouts.generate_layout_dataset()   no_head_layouts.generate_dip_layout_dataset()
        (surface_imu.surface_kinematics 或 forward_kinematics + 五点差分)
                    │                                     │
   data/no_head_5imu[_surface]_processed/<layout>/*.pt     data/no_head_5imu..._dip_test/<layout>/dip_{train,test}.pt
   {joint,pose,shape,tran,acc[N,6,3],ori[N,6,3,3],contact,layout_labels,layout_joints,synthesis}
                    │
   PoseDataset._process_file_data(): acc/ori[:, :5] 取前5槽 → forward_kinematics(局部→全局pose)
                    │
   PoseDataset._process_combo_data(): combos={"all":[0..4]} 不做mask
       → imu_input = [acc.flatten(15) ⊕ ori.flatten(45)] = 60D
       → 按 window_length(125) 切窗口 → data['imu_inputs'/'pose_outputs'/'joint_outputs'/'tran_outputs'/'vel_outputs'/'foot_outputs']
                    │
   PoseDataModule: random_split(90/10) → DataLoader(batch=256, collate=pad_seq 变长padding)
                    │
   batch: imu[B,S,60], pose[B,S,144], joint[B,S,24,3], tran[B,S,3], vel[B,S,24,3], foot[B,S,2]
                    │
   单模块训练（train.py 默认逐个训练，四者互不依赖）：
     Joints(imu 60D) → 72D 关节位置，监督信号=joint_outputs
     Poser([GT_joints+noise(0.04) ⊕ imu] 132D) → reduced pose 96D，监督信号=pose_outputs(取reduced子集)
     FootContact([GT_joints+noise(0.04) ⊕ imu] 132D) → 2D 接触概率，监督信号=foot_outputs
     Velocity([GT_joints+noise(0.025) ⊕ imu] 132D) → 72D 关节速度，监督信号=vel_outputs
                    │
   combine_weights.py 合并 4 个子模块 state_dict → base_model.pth
                    │
   联合推理（MobilePoserNet.forward()，此时才真正级联）：
     Joints(imu 60D) → pred_joints 72D
       → Poser([pred_joints⊕imu] 132D) → reduced pose → 还原全局/局部 144D
       → FootContact([pred_joints⊕imu] 132D) → 2D 接触概率
       → Velocity([pred_joints⊕imu] 132D) → 72D 关节速度
```

## 10. 与原版训练的关键差异小结

1. 传感器位置来源不同：`process.py` 用 mesh vertex 固定的 6 点（含 head），`no_head_layouts.py` 用 `LAYOUTS` 自定义的 5 点（不含 head，`waist` 代替 pelvis 作为学习位）+ 骨盆参考槽。
2. 加速度合成方式可选：`joint`（关节位置差分，理想化）或 `surface`（网格表面固定贴附点差分，更贴近真实穿戴），两者朝向合成结果相同，只有加速度不同。
3. 通过 `MOBILEPOSER_TRAIN_COMBOS=all_5imu` 关闭了原版的缺传感器组合增强，只训练「5 点全给」这一种输入模式。
4. `data.py`/模型代码本身完全没有改动——全部差异都在预处理阶段产出的 `.pt` 文件里的 `acc[:, :5]`/`ori[:, :5]`，只要 6-slot 结构对齐，下游训练代码对「标准位置」还是「自定义表面位置」是无感的。
