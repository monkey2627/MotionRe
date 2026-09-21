# 少量 IMU 下躺卧、翻滚与爬行重建：接触建模的创新空间

调研日期：2026-09-15。范围：以用户本地论文与测试报告为起点，补查公开论文；重点是少于六个物理 IMU、在线移动端计算与非常规动作。本文是定向文献综述及研究假设评估，不是穷尽检索，也不报告新模型实验结果。

**判断：这个问题仍值得研究，但“加入全身接触”“用接触抑制 root 漂移”“处理翻滚的支撑转换”都不足以单独声称新颖。更可行的切入点，是在 3–5 IMU 的不充分观测下，识别接触约束何时有效、何时应放松，并通过允许滑动和接触位置迁移的轻量估计器改善 root。是否形成论文贡献，取决于能否超越 TIP 和 GlobalPose 的已有机制，并排除基准实现问题。**

## 1. 你的实验支持什么，不支持什么

来源：[现有测试报告](../benchmark_results_analysis.md)，特别是第 4、6、7、8、9 节。以下数字是报告中的历史结果，本次未重新执行服务器测试。

| 观察 | 报告中的证据 | 可以作出的判断 |
|---|---|---|
| 姿态较接近而总体平移差距较大 | AMASS：MobilePoser / TransPose 旋转 20.07° / 19.35°，全程根误差均值 12.766 / 1.421 m | root 分支值得单独诊断；不等于已经证明接触是原因 |
| 接触动作并非一致劣化 | lying 平移 RMSE 为 0.684 / 0.708 m；transitions 为 1.130 / 1.945 m | 不能用总体均值宣称 MobilePoser 在所有地面动作都更差 |
| 全身接触证据不足 | lying 5 条，crawling 13 条；没有独立 rolling 行；只有 MobilePoser 输出统一脚滑代理 | 缺少后背、肩、肘、膝的接触评价，尤其缺少翻滚专项证据 |
| 求解失败值得研究 | PIP：104/618 序列失败，约 16.8% | 应报告失败与成功子集误差；不能只比较成功样本均值 |
| 移动端证据未闭合 | 缺参数量、设备延迟、RAM、能耗 | 运行总时长含渲染和 I/O，不能作为手机推理成本 |

还发现了会影响以上解释的**当前本地代码事实**。历史服务器是否运行同一版本，需要用 checkpoint、代码版本和日志对照；本文不把代码路径直接当成历史误差的已证实原因。

1. **DIP 没有真实平移标签。** [process.py](../base_mobileposer/mobileposer/process.py) 将 `tran` 置零并注明 `dip-imu does not contain translations`。因此该路径中的 DIP 平移指标是相对零轨迹的偏离，不能解释为真实 root 轨迹重建精度。
2. **传感器计数与消费路径不一致。** [evaluate_drift.py](../base_mobileposer/mobileposer/evaluate_drift.py) 的 `prepare_imu` 取 `acc[:, :5]`、`ori[:, :5]`，再选 3/4/5 个槽；这一函数没有消费第六个骨盆槽。报告的“4/5/6 物理 IMU＝3/4/5 学习槽＋必需骨盆参考”尚未被这条模型路径支持。需要区分采集时佩戴数量、预处理所需数量、推理实际使用数量；也不能直接重命名历史配置后宣称更少传感器优势。
3. **时间基需统一。** 共享 AMASS 预处理为 30 Hz；[TransPose/net.py](../TransPose/net.py) 将预测速度固定除以 60，PIP/PNP 的物理参数含 `delta_t=0.0166667`。必须核对桥接是否重采样，不能将时间尺度误差解释为接触缺陷。
4. **MobilePoser 存在需隔离的状态与输出路径。** [models/net.py](../base_mobileposer/mobileposer/models/net.py) 的 `forward` 调用有状态的 `self.velocity.forward_online`，但顶层 `reset` 没有清空该子模型状态；可选物理优化计算了 `tran_opt`，最终却仍返回原 `tran`。这些都需要独立检查，不能预先认为报告中的平移已经经过完整物理校正。
5. **PNP 零异常不等于约束始终有效。** [PNP/dynamics.py](../PNP/dynamics.py) 在 QP 不可行或解异常大时，会去掉 `Gx <= h` 后重求解；其中包括接触相关不等式。因此除异常退出率外，应记录“放弃约束”的回退率。
6. **输入噪声、角速度坐标、初始化也需对照。** 当前 MobilePoser/PNP drift 路径有合成噪声，TransPose/PIP bridge 没有对应的同位置加噪步骤；PNP bridge 的角速度转换与原生测试不同；PIP bridge 使用首帧 GT 姿态初始化。历史命令是否调整过这些条件尚未确认。

以上是实验解释边界，不是本文实施的修复。原测试报告与模型源码均未修改。

## 2. 最接近的先行工作已经做到哪一步

这里优先看论文的方法与局限原文。局限只归于相应论文，不能推广为整个领域都没有解决。

| 工作 | 已有机制与输入条件 | 对创新定位的影响 |
|---|---|---|
| **TIP，2022** [R3] | 六 IMU；预测 stationary body points（SBP），实现覆盖双手、双足、骨盆；§3.2 明确讨论翻滚时 SBP 沿后背与骨盆迁移；soft-IK 缓和启停跳变 | 多部位接触、滚动时接触迁移、软化切换均有直接先行工作。不能把 TIP 简化成“固定足部接触” |
| **PIP，2022** [R4] | 六 IMU；§3.2.3 非足关节距地面小于 0.5 cm 时也纳入接触；每处接触有四点支撑、摩擦与防滑约束 | “物理模型只允许双脚接触”不准确。可研究的是几何触发、关节代理和接触速度假设的适用边界 |
| **PNP，2024** [R5] | 六 IMU；非惯性坐标系的虚拟加速度建模；全局阶段沿用 PIP 式物理优化 | 不应将非惯性补偿、物理优化重新包装为新贡献；可作为强基线 |
| **GlobalPose，2025** [R6] | 六 IMU；候选双手、双足、骨盆；寻找能解释 root 残余力的最小支撑集合；结合静止概率、速度校正与确认机制 | 已覆盖非足支撑、承重变化、概率信号与平移校正，是最重要直接对手 |
| **SAIP，2025** [R7] | 六 IMU；针对体型变化建模；§4.4 明确说明非足接触的 crawling、rolling 仍困难 | 支持研究动机，但不能用这篇的局限覆盖 TIP 或其他方法 |
| **DiffusionPoser，2024** [R8] | 可变 IMU 数量与位置，允许压力鞋垫；表示包含足跟/足趾接触，进行 root 校正；正文有 2/3/4 IMU 配置比较 | “可变少传感器＋接触＋root”已有先例；生成式合理性不等于真实动作恢复 |
| **MobilePoser，2024** [R9] | 原论文为 1–3 个消费设备，融合足接触与网络速度；报告 iPhone 15 Pro 60 FPS | “少传感器＋手机实时”也不是新命题。原论文配置与本地修改后的 benchmark 路径须分开 |
| **EnvPoser，2025** [R16] | 头、手的三点 6DoF 跟踪与预扫描场景；多假设/不确定性、接触与环境几何约束 | 可借鉴接触表示，不能作为同等信息量的纯三 IMU 基线：6DoF 跟踪提供位置，IMU 不直接提供位置 |

关键原文定位：Jiang 等（2022），TIP §3.2–3.3、附录 F；Yi 等（2022），PIP §3.2.3；Yi 等（2025），GlobalPose §3.2–3.4、§4.4；Yin 等（2025），SAIP §4.4。<!--ref:tip2022--><!--anchor:section:3.2--><!--ref:pip2022--><!--anchor:section:3.2.3--><!--ref:globalpose2025--><!--anchor:section:4.4--><!--ref:saip2025--><!--anchor:section:4.4-->

两处尤其重要：

> TIP： “rolling on the floor motion results in SBPs moving across the lower back and pelvis”

> GlobalPose： “sliding contacts or those with very slight forces cannot be accurately modeled”

它们说明真正需要比较的是**接触模式、接触约束是否可靠，以及较少传感器时的估计能力**。不能只把接触候选从脚换成背部，再声称首次处理翻滚。

## 3. 本次新增的文献及其影响

下列工作不在本次枚举的 `papers/` 文件清单中。网页摘要和四篇相关全文已读取，来源文本存于 `sources/`；不意味着此前用户一定没有读过它们。

| 新查工作 | 来源与阅读深度 | 为什么需要关注 | 尚不能据此得出的结论 |
|---|---|---|---|
| **Ground Reaction Inertial Poser（GRIP）**，Hori 等，2026 [R17] | [arXiv](https://arxiv.org/abs/2603.16233)，全文；[CVF 官方条目](https://openaccess.thecvf.com/content/CVPR2026/html/Hori_Ground_Reaction_Inertial_Poser_Physics-based_Human_Motion_Capture_from_Sparse_CVPR_2026_paper.html)确认 CVPR 2026 | 四 IMU（双腕、双脚）＋鞋垫压力；KinematicsNet＋DynamicsNet，通过物理仿真重建；提供 PRISM 数据集 | 不是四个纯 IMU 的等信息量结果，也未从所读实验确认系统性翻滚测试或手机运行 |
| **IMU-HOI**，Lin 等，CVPR 2026 [R18] | [CVF 官方条目](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_IMU-HOI_A_Symbiotic_Framework_for_Coherent_Human-Object_Interaction_and_Motion_CVPR_2026_paper.html)，[全文](https://arxiv.org/html/2606.28604v1) | 身体与物体 IMU；概率手物接触，融合 FK 和惯性分支；校准接触概率；可接入已有骨干 | “概率接触＋可插拔融合”已不够新。其 §5 明确局限为准刚性单接触，未显式处理滑动、多同时接触或可变形物体 |
| **Seeing Without Eyes / IMU-to-4D**，Hsu 等，2026 [R19] | [arXiv](https://arxiv.org/abs/2604.21926)，全文；本次按预印本引用 | 少至三 IMU，LLM 联合预测动作、文字与粗场景 | 不能声称首次用纯 IMU 建模人体与环境。补充 S7 承认轨迹漂移、场景歧义与预测物体互穿；这不是其已验证人体穿地指标 |
| **MARIO**，Li 等，2026 [R20] | [arXiv](https://arxiv.org/abs/2606.02996)，全文；元数据标注 CVPR 2026 Findings | IMU 推断的人体姿态用于惯性里程计，并融合气压计、磁力计、第二 IMU | “用姿态先验改进 root”已有先行路线。315 FPS 是 A40 GPU 报告值，不能换算成手机实时 |
| **Contact-Aided Invariant EKF**，Hartley 等，2018 [R21] | [arXiv](https://arxiv.org/abs/1805.10410)，摘要及理论结论核对 | IMU＋接触的状态估计与可观测性；为 root 校正提供理论参照 | 机器人具有编码器/更可靠的运动学状态，不能直接套到少 IMU 人体的隐关节估计 |
| **Whole-Body Human Kinematics…Contact-Aided Lie Group Kalman Filter**，Ramadoss 等，2022 [R22] | [arXiv](https://arxiv.org/abs/2205.07835)，摘要 | 人体惯性＋力矩传感、压力中心接触检测、浮动基座滤波 | 接触辅助人体 root 滤波并非空白；该输入条件不同于 3–5 个纯 IMU |
| **LSTM-Based Zero-Velocity Detection**，Wagstaff 与 Kelly，2018 [R27] | [arXiv](https://arxiv.org/abs/1807.05275)，摘要与检索记录 | 学习式零速检测接 EKF，原摘要已经包含 crawling 和 ladder climbing | “神经网络检测爬行零速”本身不新，但它不等价于全身姿态重建 |
| **Inertial Human Motion Capture: From Biomechanics…**，Kok 等，2026 [R28] | [arXiv](https://arxiv.org/abs/2607.16000)，摘要；预印本综述 | 从运动条件、环境、体型与传感器可靠性梳理方法选择 | 本次未全文评估，不能拿它证明具体方法在翻滚上失败 |
| **Probabilistic Inertial Poser（ProbIP）**，Kim 等，ICCV 2025 [R29] | [CVF 官方摘要](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Probabilistic_Inertial_Poser_ProbIP_Uncertainty-aware_Human_Motion_Modeling_from_Sparse_ICCV_2025_paper.html)，另有 Crossref 元数据匹配 | RU-Mamba 预测旋转的 matrix Fisher 分布与不确定性，采用 Progressive Distribution Narrowing | 进一步排除“首次对稀疏 IMU 姿态建模不确定性”。摘要不足以判断其接触/root 功能，应在正式立项前阅读全文 |
| **MagShield**，Shao 等，ICCV 2025 [R30] | [CVF 官方摘要](https://openaccess.thecvf.com/content/ICCV2025/html/Shao_MagShield_Towards_Better_Robustness_in_Sparse_Inertial_Motion_Capture_Under_ICCV_2025_paper.html) | 多 IMU 联合检测磁干扰，再用运动先验纠正朝向，可接入已有系统 | 磁干扰检测与校正已有专门方法；不应把一般传感器扰动处理当成本次接触模块的独立创新 |

GRIP 的补充审视：其正文说明高动态转换可能使仿真人体失衡，因此加入 fall recovery；PRISM 的训练/测试为序列随机划分，不能直接当成受试者独立泛化证明。这也提醒我们：恢复真实跌倒/躺卧的估计器，需要避免把所有落地状态都当成应该纠正的控制失败。后一条是本报告提出的实验风险，而非 GRIP 已经失败的实测结论。[R17，§3、§4.1]

## 4. 建议优先推进的研究假设

以下均为本次提出的候选，不是已验证方法，也不是“首次”声明。

### 候选 A：允许滑动与接触位置迁移的稀疏 IMU root 校正

**问题表述：能否在 3–5 IMU 下，利用全身表面接触的模式与可靠性，降低地面动作中的 root 漂移，同时避免把真实滑动或翻滚锁死？**

先固定平面地面，研究背/侧躯干、肩、肘、膝、手、足等表面区域。一个小型因果模型预测：

- 接触可能性；
- 接触属于近似粘着还是允许切向滑动；
- 当前接触表面位置或少量候选位置的权重；
- 这些量的误差尺度，以及接触约束与 IMU/运动学是否相容。

滚动可以表现为一连串局部近似无滑动接触，但参与接触的材料点持续改变；不一定要将“rolling”设成一个独立动作类别。动作名称只用于分组评测，不提供躺卧/康复动作模板。

以身体表面材料坐标 `a` 表示点，世界位置为：

\[
x(a,t)=p(t)+R(t)s(a,\theta(t),\beta).
\]

其中 `p,R` 为 root，`θ` 为关节姿态，`β` 为体型。约束中使用的是**固定材料点的瞬时速度**：

\[
v_{\rm mat}(a,t)=\dot p+\omega\times Rs+RJ_\theta\dot\theta.
\]

如果滚动时选择的接触位置是 `a(t)`，沿选择位置求导还会出现 `R(∂s/∂a)·ȧ`。它表示“接触位置在身体表面迁移”，不能误当成同一个材料点的实际滑移速度。这是实现中必须区分的两个量；公式来自上述运动学定义的链式求导，不作为新的物理定律或新颖性贡献。

相应约束可分为：

| 接触状态 | 合理约束 | 应避免的错误 |
|---|---|---|
| 持续粘着，固定环境 | 当前接触材料点相对环境速度近零 | 把附近整个关节中心锁死 |
| 稳定滑动 | 法向接触/非穿透；切向允许非零速度 | 所有接触一律零速 |
| 滚动 | 约束当前接触材料点，允许接触位置随时间迁移 | 长期固定同一表面顶点；对整块躯干强加静止 |
| 离地、碰撞瞬间、状态不确定 | 减弱持续接触约束，保持运动与不确定性传播 | 为压低漂移强行吸附到地面 |

首版只修正 root 的三维速度与高度，不同时重优化所有关节。固定候选点与权重时，可用小规模加权最小二乘融合网络速度先验、法向约束、可靠的切向约束。约束冲突时使用软惩罚与有限回退；这种设计仅意味着计算规模可控，**不保证完整动力学可行，也不保证永不穿透**。

**与最接近工作的区别必须落到实验：**

- 对 TIP：不能以“能移动接触点”区分，TIP 已有；需要证明显式滑动处理、更多躯干接触区域、错误约束抑制在少 IMU 下有额外收益。
- 对 GlobalPose：不能以“支撑选择/置信度”区分；需要证明对滑动、轻接触、五候选以外的接触能改善，而不是扩大候选集合就产生全部收益。
- 对 IMU-HOI：不能以“接触概率加权”区分；需要证明多处人体地面接触与模式变化的必要性，其准刚性手物接触融合不能直接获得同等结果。
- 对 GRIP：不能只展示四 IMU。纯 IMU 与额外压力输入应分组；比较时明确是否需要推理时仿真器、环境模型和压力测量。

**新颖性风险：中高；问题价值：明确；建议优先级：最高。**它最适合成为主线，但必须由对照实验证明机制贡献。[R3、R6、R17、R18]

### 候选 B：以约束有效性和可观测性指导更新

**问题表述：减少传感器后，模型能否知道哪些 root 分量由当前接触可靠约束，并在证据不足时避免过度修正？**

接触概率高并不等于切向零速可靠。建议区分 `P(contact)` 与 `P(no-slip | contact)`，并估计接触残差协方差。参考接触辅助滤波的思想，对当前有效观测 Jacobian 的秩或奇异值进行诊断；退化方向依赖运动先验并保留较大不确定性，而非输出过强的确定性约束。[R21、R22]

限定条件必须写清：只调整 root 三维速度且姿态完全已知时，一个可靠粘着点就可能提供完整速度信息；若同时估计姿态、接触位置、IMU 偏置或滑移，问题会更退化。因此不能简单声称“接触点越多，root 的所有自由度都可观测”。

无外部锚点时，IMU 与接触一般不能确定任意绝对世界平移；没有可靠航向参考时还存在全局 yaw 的规范自由度。Hartley 等针对机器人 IMU＋接触系统给出明确不可观测结论；迁移到人体需重新写出状态与观测假设，不能照搬整个定理。[R21]

一个直接的反例：将一段固定姿态运动整体叠加恒定水平速度，理想 IMU 加速度和朝向不变；如果允许身体在地面滑动，地面高度也不变。没有额外锚点或正确的粘着假设，模型不能普遍区分它与静止。这是模型假设下的反例，说明“纯 IMU 对任意运动无漂移”不应成为目标。

训练可增加“错误施加约束的代价”，检验预测不确定性是否真的对应校正误差；用 coverage–risk 曲线、置信区间覆盖率或概率校准评价，而不只可视化权重。协方差不能通过任意增大来逃避损失，需配套适当的概率目标与校准验证。

**新颖性风险：高。**TransPose 已有置信度融合，UMotion 已有不确定性融合，IMU-HOI 已有校准接触门控，ProbIP 已直接预测旋转分布与不确定性。只有把可靠性明确联系到少 IMU 下接触模式误判、退化方向和可检验的误差收益，才有较强研究内容。适合作为候选 A 的第二模块，不宜仅以“uncertainty-aware”作为独立贡献。[R2、R13、R18、R21、R29]

### 候选 C：按接触模式比较传感器布局与轻量化

**问题表述：在相同手机计算预算下，3、4、5 个 IMU 的最佳布局，是否随着背部支撑、手膝支撑和翻滚发生系统性变化？**

布局候选可包含：三 IMU＝骨盆＋双前臂；四 IMU 分别增加一侧小腿/大腿或躯干；五 IMU＝骨盆＋双前臂＋双小腿。它们是待比较设计，不是推荐的已知最优布局。对于单侧第四传感器，应镜像平衡左右侧并考虑地面压迫/遮挡与佩戴舒适性。

比较 average error、每类动作的 P90/P95、约束回退率、参数量和目标设备延迟。不要只通过把六 IMU 输入补零得到少传感器结果；至少增加使用同一训练预算进行 sensor-dropout 训练的模型，必要时比较专门训练的少传感器模型。root 参考、校准与预处理消费的传感器也要计数。

**新颖性风险：高。**DiffusionPoser 已比较少传感器布局，IMUCoCo 已研究灵活位置，MobilePoser 已提供手机推理。独立的“少一两个传感器”很难成立。更适合作为候选 A/B 的实用性证据与最差动作覆盖分析。[R8、R9、R11]

## 5. 开始训练新模型前，最值得做的实验

### 5.1 先确认接触机制是否真的有改进空间

建议做一个冻结姿态/速度骨干的 oracle-contact 小实验。在少量人工检查的静躺、侧滚、连续翻滚、手膝爬行、肘膝爬行、真实滑动、地面起身序列上，保持相同输入与姿态预测，只替换 root 校正：

| 变体 | 所回答的问题 |
|---|---|
| 原始 root 分支 | 当前起点 |
| 统一地面高度与简单表面防穿透 | 改善是否仅来自更好的几何基准？ |
| 足部接触校正 | 双脚假设本身能提供多少收益？ |
| PIP 式非足几何接触 | 简单多部位约束是否已经足够？ |
| TIP 式 SBP / GlobalPose 式支撑基线 | 是否超越最接近的已知设计？ |
| GT 接触区域＋全部接触零速 | 理想识别能否改善？错误的零速假设会不会锁死滑动/翻滚？ |
| GT 接触区域＋模式适配＋迁移 | 在几乎没有识别误差时，候选机制是否仍有收益？ |
| 预测接触＋模式适配＋迁移 | 收益能否在真实可用输入下保留？ |

GT/oracle 仅用于诊断上限，不能放入正常方法主表。若固定预测姿态导致几何已经严重错误，可再用 GT 姿态做一个纯 root 子问题上限；这两种 oracle 必须分别标注，不能混为同一结果。

停止或调整方向的条件：

- GT 接触也不能明显改善：先检查 root 坐标/时间尺度、姿态误差、地面/体型失配，而不是直接扩大接触网络。
- GT 接触有益但预测接触无益：重点转向可辨识性、传感器布局与标注，而非更复杂求解器。
- 简单多部位几何基线与新模块相当：不足以支持复杂机制的贡献，缩减方案。
- 平均误差下降却锁死真实滑动、抑制真实跌倒或造成支撑切换尖峰：候选尚未解决目标问题。

### 5.2 数据与标签

AMASS 可用于合成训练与预实验，但运动学网格不是力学真值。低高度＋低速可产生接触伪标签，不能自动等同真实承重/压力标签；滑动接触也不能被“低速”条件排除。静止悬空点必须作为反例。[R3 附录 F；R17 输入与数据设计]

使用整段一致的地面坐标和经过检查的表面区域。不要每帧把最低关节重定义为地面，否则可能掩盖 root 高度漂移和穿透。训练标签可使用 GT，推理的地面高度应来自部署时允许的校准/估计；两个协议须分别报告。

最终需要真实少 IMU 与独立位置真值的地面动作测试。光学动捕/视频或压力设备可以用于采集 GT，但不作为纯 IMU 推理输入。按受试者、来源序列划分，防止同一原始动作的窗口同时出现在训练与测试。动作模板、康复模板和模板衍生特征均不进入方法。

首轮覆盖至少应包含静躺但四肢活动、真实滑动、连续侧滚、手膝/肘膝爬行、起身与落地、静止悬空肢体、短暂离地。样本数量应根据试点方差再确定，现有 5 条 lying 不适合做广泛稳健性结论。

### 5.3 指标与公平比较

- **姿态与全局运动：**旋转误差、root-aligned MPJPE、global MPJPE、短时 root RPE、终点漂移及每条序列误差；水平与垂直分开。
- **穿透：**固定真实地面下的全身表面穿透深度、P95、穿透帧率；同时报告 GT 网格自身与地面的拟合残差。
- **接触：**按部位统计 precision/recall；仅在真实粘着段计算不应发生的滑移；滑动段评价切向速度误差，不能把所有非零切向速度算失败。
- **滚动：**当前材料点的接触残差、接触区域迁移误差、支撑转换附近 root 速度/加速度尖峰；不对跨帧不同顶点的位移直接命名为滑移。
- **可靠性：**P90/P95、灾难性失败率、QP/约束回退率、错误接触导致的校正幅度；门控拒绝率须和误差一起报告，防止永远关闭模块取巧。
- **移动端：**同一手机、batch=1、预热后中位/P95 延迟、真实使用的历史/未来帧、参数量、FP16/INT8 体积与误差、RAM、能耗；区分模型推理、滤波、完整端到端成本。

采用同一物理 IMU 数的比较检验机制，同一计算预算的比较检验部署意义；六 IMU 强方法作为性能参照。多模态方法单列表格，保留压力/UWB/6DoF/场景输入信息。使用逐序列或逐受试者的配对统计与 bootstrap，不能把强相关帧当独立样本。

移动端可以设立“完整系统支持 30/60 Hz”的工程目标，但这是待验证预算。小型接触头和低维求解只提供轻量化的可能性，不构成已经满足手机指标的证据。

## 6. 文献证据矩阵与阅读索引

质量标记按计算机视觉/机器人领域解释：**M**＝直接方法原文，足以核对其机制；**A**＝只核对摘要/元数据，不能支撑细节或否定性覆盖判断；**P**＝本次仅确认预印本身份，不能冒充已发表成果。M/A 是阅读与主张支持程度，不是论文影响力排名。算法对照研究通常对应技术领域的比较实验，不套用临床 RCT 等级。

| 编号 | 文献（作者—年份—标题/来源） | 本次注释与证据范围 |
|---|---|---|
| R1 | Huang, Y., Kaufmann, M., Aksan, E., Black, M. J., Hilliges, O., & Pons-Moll, G. (2018). *Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time*. ACM TOG. [原文](../../papers/01_DIP-IMU_2018.pdf) | 六 IMU 与学习式姿态基线；核对摘要，A |
| R2 | Yi, X., Zhou, Y., & Xu, F. (2021). *TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors*. ACM TOG. [原文](../../papers/02_TransPose_SIGGRAPH2021.pdf) | 支撑脚与网络速度的置信度融合；摘要与本地实现，M |
| R3 | Jiang, Y., Ye, Y., Gopinath, D., Won, J., Winkler, A. W., & Liu, C. K. (2022). *Transformer Inertial Poser: Real-time Human Motion Reconstruction from Sparse IMUs with Simultaneous Terrain Generation*. SIGGRAPH Asia. [原文](../../papers/03_TIP_SIGGRAPHAsia2022.pdf) | 非足 SBP、滚动接触迁移、soft-IK；§3.2–3.3/附录 F，M |
| R4 | Yi, X., Zhou, Y., Habermann, M., Shimada, S., Golyanik, V., Theobalt, C., & Xu, F. (2022). *Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors*. CVPR. [原文](../../papers/04_PIP_CVPR2022.pdf) | 非足几何接触与物理优化；§3.2.3/§5，M |
| R5 | Yi, X., Zhou, Y., & Xu, F. (2024). *Physical Non-inertial Poser (PNP): Modeling Non-inertial Effects in Sparse-inertial Human Motion Capture*. ACM TOG. [原文](../../papers/16_PNP_SIGGRAPH2024.pdf) | 非惯性建模；PIP 式全局优化，§3.1.2/§4.4，M |
| R6 | Yi, X., Pan, S., & Xu, F. (2025). *Improving Global Motion Estimation in Sparse IMU-based Motion Capture with Physics*. ACM TOG. [原文](../../papers/25_GlobalPose_SIGGRAPH2025.pdf) | 支撑选择、漂移校正；滑动/轻接触局限，§3–4，M |
| R7 | Yin, L., Shi, Z., Wu, Y., Yi, X., Xu, F., & Guo, S. (2025). *Shape-aware Inertial Poser: Motion Tracking for Humans with Diverse Shapes Using Sparse Inertial Sensors*. ACM TOG. [原文文本](../../papersNewWay/2510.17101v1.txt) | 体型建模是已有方向；§4.4 明确爬行/翻滚局限，M |
| R8 | Van Wouwe, T., Lee, S., Falisse, A., Delp, S., & Liu, C. K. (2024). *DiffusionPoser: Real-time Human Motion Reconstruction From Arbitrary Sparse Sensors Using Autoregressive Diffusion*. CVPR. [原文](../../papers/10_DiffusionPoser_CVPR2024.pdf) | 配置灵活、足部接触/root 校正、计算成本；§3–5，M |
| R9 | Xu, V., Gao, C., Hoffmann, H., & Ahuja, K. (2024). *MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices*. UIST. [原文](../../papers/15_MobilePoser_UIST2024.pdf) | 1–3 消费设备与手机推理已存在；摘要/引言＋本地实现，M |
| R10 | Zhang, Y., Xia, S., Chu, L., Yang, J., Wu, Q., & Pei, L. (2024). *Dynamic Inertial Poser (DynaIP): Part-Based Motion Dynamics Learning for Enhanced Human Pose Estimation with Sparse Inertial Sensors*. CVPR. [原文](../../papers/11_DynaIP_CVPR2024.pdf) | 分部动力学、伪速度与真实数据泛化；摘要，A |
| R11 | Zhou, H., Arakawa, R., Agarwal, Y., & Goel, M. (2025). *IMUCoCo: Enabling Flexible On-Body IMU Placement for Human Pose Estimation and Activity Recognition*. UIST. [原文](../../papers/28_IMUCoCo_UIST2025.pdf) | 数量/佩戴位置灵活性已有先例；摘要，A |
| R12 | Armani, R., Qian, C., Jiang, J., & Holz, C. (2024). *Ultra Inertial Poser: Scalable Motion Capture and Tracking from Sparse Inertial Sensors and Ultra-Wideband Ranging*. SIGGRAPH. [原文](../../papers/17_UIP_SIGGRAPH2024.pdf) | IMU＋UWB 距离，不能等同纯 IMU；摘要，A |
| R13 | Liu, H., Ota, H., Wei, X., Hirao, Y., Perusquía-Hernández, M., Uchiyama, H., & Kiyokawa, K. (2025). *UMotion: Uncertainty-driven Human Motion Estimation from Inertial and Ultra-wideband Units*. CVPR. [原文](../../papers/23_UMotion_CVPR2025.pdf) | 六节点 IMU/UWB 与 UKF 不确定性融合；摘要，A |
| R14 | Xue, Y., Jiang, J., Armani, R., Hollidt, D., Liao, Y.-C., & Holz, C. (2025). *Group Inertial Poser: Multi-Person Pose and Global Translation from Sparse Inertial Sensors and Ultra-Wideband Ranging*. ICCV. [原文](../../papers/29_GroupInertialPoser_ICCV2025.pdf) | 每人六节点、跨人距离与平移；摘要，A |
| R15 | Hollidt, D., Bendinelli, T., & Holz, C. (2026). *Ultra Diffusion Poser: Diffusion-Based Human Motion Tracking From Sparse Inertial Sensors and Ranging-Based Between-Sensor Distances*. [arXiv:2606.02153](https://arxiv.org/abs/2606.02153)，[本地原文](../../papers/30_UDP_CVPR2026.pdf) | 六 IMU/UWB；距离引导 root 梯度为零；§3.2.4/§5，M；本地文件名不是录用证明 |
| R16 | Xia, S., Zhang, Y., Su, Z., Zheng, X., Lv, Z., Wang, G., Zhang, Y., Wu, Q., Chu, L., & Pei, L. (2025). *EnvPoser: Environment-aware Realistic Human Motion Estimation from Sparse Observations with Uncertainty Modeling*. CVPR. [原文](../../papers/24_EnvPoser_CVPR2025.pdf) | 三点有位置跟踪＋场景；摘要、引言及补充，M |
| R17 | Hori, R., Song, J.-T., Luo, Z., Cao, J., Shin, S., Saito, H., & Kitani, K. (2026). *Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors*. CVPR, 28435–28445. [原文](https://arxiv.org/abs/2603.16233)，[项目页](https://ryosukehori.github.io/grip-project/) | 四 IMU＋压力/仿真，PRISM；§3–5，M；CVF 官方条目与项目页确认发表 |
| R18 | Lin, L., Xia, S., Lai, Z., Sun, L., Yang, J., & Pei, L. (2026). *IMU-HOI: A Symbiotic Framework for Coherent Human-Object Interaction and Motion Capture via Contact-Conscious Inertial Fusion*. CVPR, 42901–42910. [原文](https://arxiv.org/abs/2606.28604) | 概率接触、分支融合、滑动局限；§3/§5，M；CVF 条目确认发表 |
| R19 | Hsu, H.-Y., Cheng, T., Wen, J., Schwing, A. G., & Wang, S. (2026). *Seeing Without Eyes: 4D Human-Scene Understanding from Wearable IMUs*. [arXiv:2604.21926](https://arxiv.org/abs/2604.21926) | 三 IMU 与联合场景生成；补充 S7 的漂移/歧义，M/P |
| R20 | Li, Y., Yeon, T., Gao, C., Xu, V., Liu, X., & Ahuja, K. (2026). *MARIO: Motion-Augmented Real-Time Multi-Sensor Inertial Odometry*. [arXiv:2606.02996](https://arxiv.org/abs/2606.02996) | 姿态引导里程计与辅助传感；实现/局限，M；元数据为 CVPR 2026 Findings，不写成主会论文 |
| R21 | Hartley, R., Ghaffari Jadidi, M., Grizzle, J. W., & Eustice, R. M. (2018). *Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State Estimation*. [arXiv:1805.10410](https://arxiv.org/abs/1805.10410) | 接触辅助状态估计与绝对位置/yaw 不可观测性；摘要支持，A；基础理论，不作人体性能证据 |
| R22 | Ramadoss, P., Rapetti, L., Tirupachuri, Y., Grieco, R., Milani, G., Valli, E., Dafarra, S., Traversaro, S., & Pucci, D. (2022). *Whole-Body Human Kinematics Estimation using Dynamical Inverse Kinematics and Contact-Aided Lie Group Kalman Filter*. [arXiv:2205.07835](https://arxiv.org/abs/2205.07835) | 惯性＋力矩的全身/基座估计；摘要，A；本次不另断言发表身份 |
| R23 | Wu, Y., Wang, C., Yin, L., Guo, S., & Qin, Y. (2024). *Accurate and Steady Inertial Pose Estimation through Sequence Structure Learning and Modulation*. NeurIPS. [原文文本](../../papersNewWay/NeurIPS-2024-accurate-and-steady-inertial-pose-estimation-through-sequence-structure-learning-and-modulation-Paper-Conference.txt) | 序列结构与稳定姿态；§10 明确缺全局平移，M |
| R24 | Wu, Y., Guo, S., & Qin, Y. (2025). *MODA: Motion-Drift Augmentation for Inertial Human Motion Analysis*. CVPR. [原文文本](../../papersNewWay/Wu_MODA_Motion-Drift_Augmentation_for_Inertial_Human_Motion_Analysis_CVPR_2025_paper.txt) | 佩戴位移的 IMU 一致性增强；“drift”不应自动解释为 root 轨迹漂移，摘要/方法检索，A |
| R25 | Zuo, C., Huang, J., Jiang, X., Yao, Y., Shi, X., Cao, R., Yi, X., Xu, F., Guo, S., & Qin, Y. (2025). *Transformer IMU Calibrator: Dynamic On-body IMU Calibration for Inertial Motion Capture*. ACM TOG, 44(4). [原文文本](../../papersNewWay/TIC_camera_ready.txt) | 动态标定与运动多样性触发已有先例；首页与摘要，A |
| R26 | Zhang, L., Yi, X., & Xu, F. (2025). *BaroPoser: Real-time Human Motion Tracking from IMUs and Barometers in Everyday Devices*. UIST. [原文](../../papers/27_BaroPoser_UIST2025.pdf) | 两个设备的 IMU＋气压与大腿参考；低传感器和高度辅助路线，摘要/引言，M |
| R27 | Wagstaff, B., & Kelly, J. (2018). *LSTM-Based Zero-Velocity Detection for Robust Inertial Navigation*. IPIN. [arXiv:1807.05275](https://arxiv.org/abs/1807.05275) | 爬行下学习零速＋EKF 先行工作；摘要，A |
| R28 | Kok, M., Weygers, I., Osman, H., Weber, D., Li, R., Seel, T., & Seth, A. (2026). *Inertial Human Motion Capture: From Biomechanics to Recent Sensor Fusion Methods and Back*. [arXiv:2607.16000](https://arxiv.org/abs/2607.16000) | 方法选择与可靠性的教程综述；摘要，A/P |
| R29 | Kim, M., Jeon, Y., & Jo, S. (2025). *Probabilistic Inertial Poser (ProbIP): Uncertainty-aware Human Motion Modeling from Sparse Inertial Sensors*. ICCV, 25893–25902. [官方来源](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Probabilistic_Inertial_Poser_ProbIP_Uncertainty-aware_Human_Motion_Modeling_from_Sparse_ICCV_2025_paper.html) | 旋转 matrix Fisher 分布、RU-Mamba、PDN；官方摘要，A；Crossref DOI 10.1109/ICCV51701.2025.02402 元数据匹配 |
| R30 | Shao, Y., Yi, X., Yin, L., Guo, S., Yong, J., & Xu, F. (2025). *MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances*. ICCV, 29021–29030. [官方来源](https://openaccess.thecvf.com/content/ICCV2025/html/Shao_MagShield_Towards_Better_Robustness_in_Sparse_Inertial_Motion_Capture_Under_ICCV_2025_paper.html) | 磁干扰的 detect-then-correct 与可插拔校正；官方摘要，A |

补充筛查但不纳入主证据链：FIP 为四 IMU＋两肘部弯曲传感器，主要处理衣物位移；SATPose 是单目＋压力相关路线；RobustCap 使用单目＋IMU；AnyMo（arXiv:2605.22715）主要评价活动识别、检索和动作描述。它们真实存在并有相关启发，但本次不用于声称地面接触 root 机制的性能或新颖性。未核验存在性的检索线索不列入参考文献。

## 7. 检索与核验记录

- **本地起点：**`papers/` 31 个 PDF 文件，包含编号 01–30 及额外 IMUCoCo 文件；同时查看 `papersNewWay/` 中与体型、标定、增强、压力感知相关资料，以及 `code/benchmark_results_analysis.md`。PDF 文件数量不是独立论文数量，也不是全文精读数量。
- **范围：**2018–2026 年重点相关工作；早期零速/接触估计用于追溯先行技术。不筛掉否定候选新颖性的工作。
- **纳入：**稀疏 IMU 姿态/root、非足接触/滑动/滚动、接触辅助滤波、少传感器与移动部署、提供额外位置/接触观测的直接对照。只读摘要的条目明确标记，不靠摘要缺失来断言没有某模块。
- **排除主链：**纯活动识别、流体/软体机器人等同词异义命中、无可核对原始来源的条目；视觉/场景/压力融合可以作异信息量参照，但不算纯 IMU 基线。
- **公开检索：**arXiv `inertial human motion`、`inertial contact human`、`inertial rolling`、`inertial crawling`、`contact slip human IMU`，按最新时间排序；跟踪原文引用补查接触滤波/概率方法。首个宽检索只取前 50 条；因此不能宣称穷尽覆盖。
- **来源核对：**原始 PDF/HTML 标题、作者、方法/局限；IMU-HOI、GRIP 另核对 CVF 官方条目，GRIP 也核对作者项目页；ProbIP 通过 Crossref 与 ICCV 官方目录/摘要核对，MagShield 通过 ICCV 官方目录/摘要核对。ProbIP 的 arXiv 查询无命中，最初构造的 CVF 路径因大小写不符返回 404，随后从官方目录取得正确路径；DOI 解析返回空 202 页面，不作为解析成功证据。预印本与会议版本按标题/作者/编号去重，版本年份与发表年份分开。
- **失败与降级：**arXiv 直连超时后，经本机代理成功；Google 返回访问提示页，Bing 泛查询相关性差，没有用它们作证据；Semantic Scholar API 返回 429，未获得 S2 去重结果。跨索引/DOI 解析没有完成全覆盖，因此不宣称完成系统化参考文献完整性认证，也不宣称已核验所有撤稿与利益冲突状态。
- **留痕：**[retrieve.py](retrieve.py) 保存 URL 与来源文本到 `sources/`；本地 PDF 的转换文本以 `local_` 开头。联网来源可按 URL 在该目录检索。查询实际结果与服务错误应同时理解，检索零结果不等于论文不存在。
- **覆盖偏斜：**本次是计算方法为主的技术问题检索，30 项中至少 29 项为算法/方法工作或其相关基础研究；因此缺少力学测量与临床用途证据是范围限制，不以这份综述推断临床效果。没有对全部论文两两检查矛盾。

本次检查了以下易混淆关系：

| 比较 | 张力类型与处理 | 仍需用户/后续实验确认 |
|---|---|---|
| TIP 与 SAIP | 条件不同：TIP 明确有滚动 SBP；SAIP 明确承认非足接触局限。不能拼成“所有方法不支持翻滚” | 同一真实翻滚数据下的性能 |
| GlobalPose 与 IMU-HOI | 支撑接触与手物接触任务不同；二者都限制了泛化的“概率接触融合首次”说法 | 多处滑动地面接触是否超出其有效能力 |
| GRIP 与纯少 IMU 候选 | 观测条件不同：压力是额外信息；物理合理与真实姿态精确也不是同一指标 | 相同输入下的机制收益与计算预算 |
| 原版 MobilePoser 与本地 benchmark | 协议差异尚未闭合：原版消费设备方案不能自动代表本地输入与优化分支 | 历史服务器源码、日志、模型的一致性 |

上述四组是定向检查，不是完整的两两矛盾检测；解释仍有待研究者确认。三次自查分别修正了：初始问题中“仅关节旋转”的过宽概括；分析中忽略 TIP/GlobalPose 的新颖性风险；交付前把协议混杂误当机制证据的风险。没有声称独立多审稿人认证。

## 8. 建议采用的最终定位

建议的工作题目：**面向地面动作的少量 IMU 人体重建：接触模式与约束可靠性驱动的轻量 root 估计**。

核心主张应是待检验的条件性陈述：在相同实际传感器数、采样率、噪声和因果推理预算下，允许滑动与接触位置迁移，并抑制不可靠接触约束，是否比足部方法、TIP 式 SBP 和 GlobalPose 式支撑校正更准确地恢复躺卧、翻滚、爬行及其转换。

优先做协议核对和冻结骨干的 oracle-contact 实验，再决定是否训练模式/不确定性模块。若这个实验没有清晰收益，就不应先投入复杂网络与完整论文包装。

**局限与 AI 使用说明：**本文由 AI 辅助检索、原文摘取、代码静态核查与综合撰写；两个并行只读任务分别核对关键原文和实现，不能视为独立误差过程或同行评审。没有执行新的性能实验，没有验证手机部署，没有穷尽所有 2026 工作；候选创新及其数学建模均需研究者复核，并通过公平实验建立证据。
