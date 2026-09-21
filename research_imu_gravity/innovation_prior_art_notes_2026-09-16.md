# 20 个创新候选的先行工作核验备忘

核验日期：2026-09-16。角色：有界 bibliography/source-verification。服务于 `imu_gravity_pose_disentanglement_review_2026-09-15.md` 的后续研究，不是全面新颖性检索，也不报告新实验结果。

## 范围与可追溯性

- 本轮检索本地 `papersNewWay/*.txt`、`code/research_imu_contact/sources/*.txt`，关键词覆盖 drift、calibration、uncertainty、probabilistic、placement、distillation、domain、limitations。
- 直接阅读下列论文的摘要，以及与问题相关的方法和限制段；ProbIP 和 AnyMo 只读官方网页的本地缓存。没有重新联网核实最新版本、录用状态或勘误，没有开展 Semantic Scholar ID 去重；以题目和 arXiv/DOI 去重，S2 状态未解析。
- 行号按文件中的 LF 换行计算，与 `rg -n` 一致。PDF 文本含换页字符，不能用 Python `splitlines()` 计算引用行号。双栏抽取的相邻句子有时交错，以下引文只抽取同一栏的连续语义。
- 证据分层：论文原文支持“该方法做了什么/作者报告何种局限”；它不自动证明候选机制首次出现，也不证明其局限在本项目中造成了相同错误。
- 当前集合明显集中于 2024—2026 年、稀疏惯性重建和所给本地材料；不代表整个姿态融合、机器人可观测性或移动端部署领域。

## 1. MODA 已有动态佩戴偏移增强和域泛化

**书目**：Wu, Y., Guo, S., & Qin, Y. (2025). *MODA: Motion-Drift Augmentation for Inertial Human Motion Analysis*. CVPR. [官方全文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_MODA_Motion-Drift_Augmentation_for_Inertial_Human_Motion_Analysis_CVPR_2025_paper.pdf)。本地文件：`papersNewWay/Wu_MODA_Motion-Drift_Augmentation_for_Inertial_Human_Motion_Analysis_CVPR_2025_paper.txt`。

- 第 30—39 行： “simulates the natural displacement of body-worn IMUs during motion”，并覆盖 “domain adaptation” 和 “domain generalization”，任务包括 HAR 与 HPE。
- 第 276—285 行将 IMU 相对骨骼的旋转和滑移造成的 offset 明确写成随时间变化的量。
- 第 313—330 行分别给出旋转 offset 的增量更新与滑移 acceleration offset 的更新；第 526—532 行强调同一动作下生成不同佩戴漂移信号，和改变动作本身的增强不同。
- **创新排除项**：不能把“训练时随机动态外参/佩戴滑移增强”“物理一致 IMU 增强”“利用增强做域泛化”单独作为首次贡献。须证明比 MODA 增加了何种可辨识状态、误差机制或真实非常规动作收益。
- **阅读边界**：摘要、合成方法、与动作增强的区别；未复现其全部五种学习设置。没有据此断言其已解决原始比力—重力解耦。

## 2. TIC 已有动态标定、双误差参数与活动触发；有直接相关局限

**书目**：Zuo, C., Huang, J., Jiang, X., Yao, Y., Shi, X., Cao, R., Yi, X., Xu, F., Guo, S., & Qin, Y. (2025). *Transformer IMU Calibrator: Dynamic On-body IMU Calibration for Inertial Motion Capture*. ACM TOG, 44(4). [DOI](https://doi.org/10.1145/3730937)。本地：`papersNewWay/TIC_camera_ready.txt`。

- 第 41—57 行同时估计 coordinate drift `R_G'G` 与 measurement offset `R_BS`，要求它们短窗近似不变、运动/观测足够多样，并有 “a calibration trigger based on the diversity of IMU readings”。第 12—18 行提供发表书目信息。
- 第 399—403 行说明独立计算每个 IMU 的旋转多样性，网络估计所有 IMU 参数，但只校正达到触发阈值的节点。
- **限制原文，第 637—648 行**：“large and sudden changes”；“low-activity scenarios”；“do not support the correction of global yaw drift”；“irregular motions may lead to incorrect calibration”。
- **创新排除项**：动态外参估计、把世界坐标漂移与 sensor-to-bone offset 分开、只在可信活动窗校正均非空白。新方案若针对低活动、突然滑移、不规则运动与少于六节点的校正可靠性，需把这些作为独立可证伪条件；不能承诺纯 IMU 无条件恢复全局 yaw。
- **阅读边界**：作者稿摘要、触发策略、数据采集和限制段；未完整复核其假设推导，也未复现实验。该工作还测了 MobilePoser 的消费级位置，不能概括为完全没研究消费设备。

## 3. Loose Inertial Poser 已有 4 IMU、衣物次级运动建模与合成

**书目**：Zuo, C., Wang, Y., Zhan, L., Guo, S., Yi, X., Xu, F., & Qin, Y. (2024). *Loose Inertial Poser: Motion Capture with IMU-attached Loose-Wear Jacket*. CVPR. [DOI](https://doi.org/10.1109/CVPR52733.2024.00215)。本地：`papersNewWay/Loose_Inertial_Poser_Motion_Capture_with_IMU-attached_Loose-Wear_Jacket.txt`。

- 第 28—44 行：4 个 IMU 集成于宽松夹克；“Secondary Motion AutoEncoder (SeMo-AE)” 学习和合成皮肤与衣物之间次级运动对 IMU 的影响；图 1 明确是 “real-time upper-body motion capture”。
- 第 39—40 行测试 zipped / unzipped 两种穿法。
- **创新排除项**：四个 IMU、宽松佩戴、学习衣物扰动的合成器都不是新点；不能把它的上半身结果误报为四 IMU 全身及 root 重建。可支持进一步验证全身、地面接触使衣物受力改变、非常规姿态下的误差解释，但该跨场景推断本身不是已证实缺陷。
- **阅读边界**：摘要、任务范围、问题引入；未复核全部动作覆盖和实时硬件配置。

## 4. ProbIP 已有少于六 IMU、Mamba 与旋转不确定性；SSM 缩写需辨别

**书目**：Kim, M., Jeon, Y., & Jo, S. (2025). *Probabilistic Inertial Poser (ProbIP): Uncertainty-aware Human Motion Modeling from Sparse Inertial Sensors*. ICCV, 25893–25902. [官方摘要](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Probabilistic_Inertial_Poser_ProbIP_Uncertainty-aware_Human_Motion_Modeling_from_Sparse_ICCV_2025_paper.html)。本地缓存：`code/research_imu_contact/sources/9523e318731c.txt`。

- 第 16—20 行作者、题目和摘要；明确 “RU-Mamba blocks” 预测 “a matrix Fisher distribution over rotations”，有 “Progressive Distribution Narrowing”，并测试 “six and fewer IMU sensors”。
- **创新排除项**：少 IMU 概率姿态、旋转分布、Mamba 模块、逐层收窄不确定性均已有。可以讨论特定误差归因或重力更新校准，但不能声称已有方法都输出确定性姿态。
- **阅读边界**：只读官方摘要缓存，不能推断其所用少传感器具体布局、contact/root 能力、实测手机耗时、概率校准质量。

**另一个 SSM**：Wu, Y., Wang, C., Yin, L., Guo, S., & Qin, Y. (2024). *Accurate and Steady Inertial Pose Estimation through Sequence Structure Learning and Modulation*. NeurIPS 2024. 本地 `papersNewWay/NeurIPS-2024-accurate-and-steady-inertial-pose-estimation-through-sequence-structure-learning-and-modulation-Paper-Conference.txt` 第 14—29 行：SSM 指 **Sequence Structure Module**，学习/指定固定长度序列的空间结构与时间平滑先验，第 56—58 行使用六个 IMU。它不是此文提出的 state-space model/Mamba。仅“加空间结构”和“平滑调制 Transformer”亦不能作为首次。

## 5. Motion Label Smoothing 已有骨架相关的标签正则；已比较物理教师软标签

**书目**：Meng, Z., Yin, L., Hou, Y., Chen, A., Guo, S., & Qin, Y. (2025). *Improving Sparse IMU-based Motion Capture with Motion Label Smoothing*. arXiv:2511.22288v1，2025-11-27。[原文](https://arxiv.org/abs/2511.22288v1)。本地 `papersNewWay/2511.22288v1.txt`。

- 第 27—44 行：将分类 label smoothing 引入该回归任务，通过 skeleton-based Perlin noise 保留相关运动属性；第 101—120 行涉及时间平滑、关节相关等性质。
- 第 407—416 行：比较项 “Knowledge Distillation” 使用经物理模块优化的姿态作为软目标，与 GT 混合形成 “distillation-based LSR”。
- 第 409—412 行同时说明训练动作范围有限，对 “slipping and street dance” 构成挑战。
- **创新排除项**：骨架感知低频标签噪声、一般运动标签正则，以及“把物理优化结果教给网络”都不能孤立作为全新概念。该比较不等于已经完成本文所需的六到三 IMU 条件分布蒸馏，也没有证明手机部署。
- **阅读边界**：摘要、方法目的、替代方案和限制；不据此把所有特殊动作失败归因于训练分布，也不引用其“first”措辞作为穷尽事实。

## 6. SAIP 已有形状—姿态测量分解，明确承认爬行/翻滚局限

**书目**：Yin, L., Shi, Z., Wu, Y., Yi, X., Xu, F., & Guo, S. (2025). *Shape-aware Inertial Poser: Motion Tracking for Humans with Diverse Shapes Using Sparse Inertial Sensors*. ACM TOG, 44(6). arXiv:2510.17101v1，2025-10-20。[DOI](https://doi.org/10.1145/3763311)。本地 `papersNewWay/2510.17101v1.txt`。

- 第 15—29 行：体形改变 IMU 加速度；先映射至成人体形的加速度，再恢复姿态和真实体形速度，结合 shape-aware physical optimization。
- **直接局限，第 520—527 行**：“struggles to handle ground contact beyond the feet, such as crawling or rolling on the ground”；采用平地假设，pull-ups/climbing 会被优化中的重力拉回地面。
- 第 528—533 行：磁干扰会导致 yaw misalignment。
- **创新排除项**：以体形解释加速度差异、体形—姿态分解及形状感知物理层均有先例。非足接触与地形假设有直接缺口依据，但不能将其等同于该方法“把重力当身体竖直”；其成人体形映射路线也不应照搬为本项目的新模板路线。
- **阅读边界**：摘要和限制段；未审计该文独立 GT 与完整数据分布。`papersNewWay/SAIP.pdf` 与本文件可能是同一作品，不应重复计数。

## 7. TVS 是物理模仿学习难度指标，不是 IMU 逆问题可观测性

**书目**：Meng, Z., Yin, L., Chen, X., Zuo, C., Chen, A., Guo, S., & Qin, Y. (2026). *Distinguishing Imitation Error from Intrinsic Motion Learning Difficulty*. 本地原文页眉标注 ICML 2026 / PMLR 306；arXiv:2512.07248v2，2026-06-08。[原文](https://arxiv.org/abs/2512.07248v2)。本地 `papersNewWay/2512.07248v2.txt`。本轮未联网核验最终会议元数据。

- 第 9—58 行：Torque Variation Score 衡量为纠正微小姿态扰动而需要的 torque variation，研究 UHC、PHC+ 等物理 imitation policy 的学习难度；提出 MID、DSJE 和异常动作检测用途。
- 第 606—617 行：当前 CPU/RBDL 实现处理 100 帧约需 1—2 分钟；“TVS characterizes difficulty under idealized dynamics”；sensor noise 等额外挑战未覆盖。
- **创新排除项/使用限制**：不能把“难动作不等于差模型”这个一般诊断思想宣称首次；也不能直接以 TVS 作为少 IMU 可观测性或重力可信度真值。高物理控制难度与不同姿态产生近似相同 IMU 的观测歧义是不同问题。本文可作分层评测思想参考，当前实现不适合作每帧手机在线模块。
- **阅读边界**：摘要、评价用途、限制；未重推全部理论，未核验所有证明条件。

## 8. AnyMo 已有布置视图与 masked 部分观测预训练，任务是运动理解

**书目**：Chen, B., Li, Z., Wongso, W., Li, L., Lin, X., Xue, H., Tag, B., & Salim, F. (2026). *AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild*. arXiv:2605.22715，v1 2026-05-21，缓存指向 v2。[原文](https://arxiv.org/abs/2605.22715)。本地 `code/research_imu_contact/sources/7ee510544287.txt`。

- 第 11—31 行日期、作者；第 36 行完整摘要： “physics-grounded IMU simulation over dense body-surface placements”， “paired synthetic placement views and masked partial observations” 训练 graph encoder，随后对齐 LLM。
- 同行任务是 zero-shot activity recognition、cross-modal retrieval、motion captioning。
- **创新排除项**：设备/位置无关表示、身体表面密集合成、跨布置视图和缺测掩码预训练均有直接先例；不能把运动理解能力等同于全身几何恢复或 root 倾斜准确性。若提出布局条件化重力重建，需保留几何信息，明确与 setup-agnostic 语义表示的目标差异。
- **阅读边界**：官方 arXiv 摘要的本地缓存；未读全文，不断言其任意布局均有效，也不以摘要指标与姿态模型直接排名。

## 9. UDP 已有测距几何重建及观测引导 diffusion，但额外依赖 UWB

**书目**：Hollidt, D., Bendinelli, T., & Holz, C. (2026). *Ultra Diffusion Poser: Diffusion-Based Human Motion Tracking From Sparse Inertial Sensors and Ranging-Based Between-Sensor Distances*. arXiv:2606.02153v1，2026-06-01。[原文](https://arxiv.org/abs/2606.02153v1)。本地 `code/research_imu_contact/sources/local_30_UDP_CVPR2026.txt`。本轮依据页眉记录预印本身份，未仅凭文件名断言已被 CVPR 接收。

- 第 12 行图示 “6 IMU+UWB sensors”；第 37—62 行摘要：Spatial Layout Module 从 UWB 距离解析重建三维节点位置；UWB-Diffusion Guidance 在采样时使姿态与实测距离一致。
- 第 73—80 行说明 MDS 闭式解和 in-the-loop forward kinematics；第 173 行显式 `k=6`。
- **创新排除项**：把传感器间几何作为约束而非拼接特征、用测量一致性引导生成模型均已有。少节点主动测距或无额外硬件的不同观测方程仍需单独论证；不能把节点数少于六与总传感器负担下降混为一谈，也不能把 UWB 得到的几何信息算作纯 IMU 信息。
- **阅读边界**：摘要和方法引入，未复核性能表、手机延迟及全部噪声模型。

## 10. 额外传感模态和手机实现也不能笼统包装为首次

**FIP 原文**：Zheng, R., Fang, J., Yao, Y., Gao, X., Zuo, C., Guo, S., & Luo, Y. (2025). *FIP: Endowing Robust Motion Capture on Daily Garment by Fusing Flex and Inertial Sensors*. [DOI](https://doi.org/10.1145/3706598.3714140)。本地 `papersNewWay/3706598.3714140.txt` 第 22—33 行摘要说明 4 IMU 加两个肘部 flex sensors，Displacement Latent Diffusion Model 与 Physics-informed Calibrator 处理佩戴位移。不能将其描述为总共四传感器；一般“物理约束校正佩戴扰动”有先例。

**手机部署基线**：MobilePoser 是本项目原报告和活动实现已经明确采用的先行工作；本轮未另做其全文的硬件/延迟核查，所以不重报具体毫秒数。任何轻量网络提议必须以同设备、同在线窗口和同物理节点数对比，而不是把换骨干、蒸馏、量化本身称为该领域首创。

**相关但不直接等价**：`papersNewWay/jiawei24a.txt` 第 1—32 行的 SuDA（Fang 等，2024）面向 flexible sensors 的 hinge joint tracking，用 predictive-function support alignment 做 Sim2Real；不能把其监督/传感输入条件直接移植成全身 IMU 证据。`papersNewWay/3721238.3730625.txt` 是布料/可变形体异构调度，非 IMU 姿态网络。`papersNewWay/116848.txt` 是 ToF-IP 演示材料，第 23—27 行写明保留六节点并增加 ToF；它不支持少于六 IMU 全身系统已经实现的结论。

## 交给主报告的证据使用建议

20 个候选应逐项区分“先行机制”“尚待验证的差异化假设”“最小可证伪实验”。最有直接文本支持的不足是 TIC 的不规则/低活动/突然变化标定局限、SAIP 的非足接触和地形局限，以及现有数据动作范围有限。没有直接材料支撑的具体缺口要标为推断或待检索，不能写成“现有方法从未研究”。
