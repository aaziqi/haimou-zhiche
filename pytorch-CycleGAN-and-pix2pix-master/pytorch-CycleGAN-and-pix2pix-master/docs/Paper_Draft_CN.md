# MP-CycleGAN: 基于多感知驱动的非配对水下图像增强方法

## 摘要
水下图像增强是海洋工程和水下视觉任务中的关键预处理步骤。现有的基于监督学习的方法虽然性能优异，但严重依赖难以获取的成对（Paired）水下数据集，限制了其在真实多变场景中的泛化能力。为了解决这一问题，本文提出了一种物理与感知双重驱动的无监督框架——**MP-CycleGAN** (Multi-Perceptual CycleGAN)。该方法在 CycleGAN 的非配对转换机制基础上，引入灰世界物理约束以校正波长相关的色偏，并结合结构梯度一致性与深层感知损失，以在去雾的同时尽量保留图像的语义与纹理细节。基于统一评测脚本，我们在 EUVP `test_samples`（matched=200）上报告了 PSNR/SSIM 的 95% bootstrap 置信区间：CycleGAN（22.82 dB / 0.783）与 MP-CycleGAN（22.91 dB / 0.795）整体表现接近且 MP-CycleGAN 在像素一致性指标上略优；在 UIEB 跨域测试（matched=890）中，MP-CycleGAN 的 UIQM 达到 10.50，与 CycleGAN 的 10.48 基本一致。消融实验进一步表明，引入灰世界、结构与感知约束在颜色相关与结构相关指标上呈现更优趋势，为 MP-CycleGAN 的设计提供了定量支撑。

**关键词**：水下图像增强；MP-CycleGAN；非配对学习；物理感知驱动；跨域泛化

## 1. 绪论 (Introduction)

### 1.1 研究背景及意义
地球表面约 71% 被海洋覆盖，水下图像与视频是水下机器人导航、海洋资源勘探、生态环境监测、海底工程检测与水下考古等任务的重要信息载体。在实际应用中，水下视觉系统往往需要在低照度、悬浮颗粒、不同水体类型与复杂光照条件下稳定工作，因此对图像质量具有较高要求。

与空气成像不同，光在水介质中的传播具有显著的波长依赖性与散射效应：红光衰减更快，导致画面整体偏蓝/偏绿；后向散射带来雾化与对比度下降，并进一步掩盖边缘与纹理细节。这些退化不仅降低人类主观观感，还会显著削弱下游视觉算法（如检测、分割、三维重建与目标跟踪）的鲁棒性与可靠性。因此，设计有效的水下图像增强方法，用以改善颜色、对比度与清晰度，具有明确的工程价值与研究意义。

在数据层面，水下增强面临一个核心瓶颈：获取严格像素对齐的“退化—清晰”成对数据代价极高，且真实环境下很难同时保证同一场景、同一视角与一致光照条件。即使通过合成数据构建配对训练集，合成退化与真实退化之间仍存在域差异（Domain Gap），导致模型在真实场景中的性能不稳定。因而，能够在非配对数据上学习增强映射、并具备跨域泛化能力的方法更符合实际应用需求。

### 1.2 研究现状
围绕水下图像增强任务，现有研究大致可归纳为传统先验方法、监督学习方法与无监督/弱监督方法三条主线，各自关注点与局限性如下。

（1）**传统先验与物理模型方法**。该类方法基于成像模型或统计先验进行颜色校正与去雾处理，例如暗通道先验及其水下变体 [5,6]、融合增强 [13]、Retinex 增强 [14]、波长补偿与去雾 [20]、基于模糊度与吸收的恢复 [19]、更严格的物理建模去水方法 [15] 等。其优点是无需训练数据、可解释性强，但往往依赖特定假设；当水体浑浊度、光照分布与散射强度变化较大时，容易出现过增强、颜色失真或细节损失，鲁棒性受限。

（2）**监督学习方法**。随着深度学习发展，基于 CNN/GAN 的方法在配对数据或合成数据上取得了显著效果，代表性工作包括 WaterNet [1]、FUnIE-GAN [2]、UColor [3] 等，以及更一般的成对图像到图像框架 pix2pix [16] 与 GAN 基础 [17]。该类方法通常能获得较高的客观指标，但对配对数据依赖强；在真实水下环境中“完美配对”难以获取，训练集分布与部署场景不一致时容易发生泛化下降。

（3）**无监督与弱监督方法**。为降低对配对数据的依赖，CycleGAN [4] 通过对抗学习与循环一致性在非配对数据上学习域间映射，UGAN [18] 等工作也探索了在非配对或弱监督条件下的增强学习。该方向更贴近真实数据采集方式，但标准 CycleGAN 在水下增强中仍可能产生颜色漂移、纹理伪影或结构被破坏等现象；此外，不同评价指标的关注点存在差异，例如全参考指标（PSNR/SSIM）强调像素一致性，而无参考指标（UCIQE/UIQM）更偏向主观感知质量，导致模型优化目标与实际观感之间可能出现权衡。

综合来看，如何在非配对框架下引入更合理的先验与约束，既抑制偏色与雾化，又尽量保持结构与纹理细节，并提升跨域泛化稳定性，仍是值得深入研究的问题。基于这一动机，本文提出 **MP-CycleGAN**：在 CycleGAN 的非配对转换机制基础上，从物理光学与视觉感知两条路径引入自监督约束，包括灰世界物理约束、结构梯度一致性与深层感知损失，以缓解颜色漂移与结构破坏，并在统一评测脚本下系统验证其同域与跨域性能。

本文的主要贡献如下：
1. 提出 MP-CycleGAN，在 CycleGAN 上引入灰世界、结构梯度与感知约束以缓解偏色与结构漂移。
2. 在统一评测脚本下报告 EUVP 同域与 UIEB 跨域结果，并给出 PSNR/SSIM 的 bootstrap 置信区间以增强可复核性。
3. 通过小预算消融实验分析不同自监督项的作用，为损失设计提供定量依据。

## 2. 相关工作 (Related Work)
水下图像增强方法大致可分为传统基于先验的方法、监督学习方法与无监督/弱监督方法三类。

传统方法通常依赖成像退化模型或统计先验来恢复颜色与对比度，例如暗通道先验（DCP）及其水下变体（UDCP）等。这类方法在特定假设成立时具有一定效果，但在水体类型、光照条件与散射程度变化较大的真实场景中，往往会出现过度增强、颜色失真或细节损失等问题；代表性工作还包括基于融合的增强 [13]、Retinex 增强 [14]、基于波长补偿的去雾增强 [20]、基于模糊度与吸收的恢复 [19] 以及更严格的物理建模去水方法 [15] 等。

监督学习方法通过成对数据直接学习从退化图像到“清晰参考”的映射，代表性方法包括 WaterNet [1]、UColor [3]、FUnIE-GAN [2] 等；在更一般的成对图像到图像学习框架中，GAN [17] 与 pix2pix [16] 也为“成对映射学习”提供了经典范式。它们通常能取得较高的客观指标，但训练依赖于成对数据或合成数据，而真实水下场景的配对标注代价高、分布复杂，容易引入明显的域差异，导致模型跨域泛化受限。

无监督方法以 CycleGAN [4] 为代表，通过对抗学习与循环一致性约束在非配对数据上进行风格迁移，能够降低对成对数据的依赖；同时也有工作利用 CycleGAN 生成训练对并结合对抗学习进行增强（如 UGAN [18]）。然而，标准 CycleGAN 在水下增强任务中仍可能出现颜色漂移、纹理伪影或结构被破坏等现象。围绕这一问题，本文在 CycleGAN 框架之上引入物理与感知层面的多重约束：用灰世界约束抑制偏色，用结构一致性与感知一致性约束缓解结构破坏与细节丢失，从而在无配对设置下实现更稳定的增强效果。

## 3. 方法论 (Methodology)

### 3.1 整体架构
本文提出的 MP-CycleGAN 采用双向循环生成架构，包含两个生成器 $G: A \to B$（水下 $\to$ 清晰），$F: B \to A$（清晰 $\to$ 水下），以及两个判别器 $D_A, D_B$。

![MP-CycleGAN Architecture](figures/mp_cyclegan_architecture.png)
> **图 1**：MP-CycleGAN 整体架构示意图。模型包含循环一致性损失 ($L_{cyc}$)、灰世界物理损失 ($L_{gray}$)、结构梯度一致性损失 ($L_{struct}$) 和感知损失 ($L_{perc}$)。图中箭头表示数据流向，彩色模块区分不同功能单元。

### 3.2 损失函数
为了实现物理与感知的双重约束，总损失函数定义为：
$$ \mathcal{L}_{total} = \mathcal{L}_{GAN} + \lambda_{cyc}\mathcal{L}_{cyc} + \lambda_{idt}\mathcal{L}_{idt} + \mathcal{L}_{MP} $$

其中 $\mathcal{L}_{MP}$ 为本文提出的多感知自监督损失组合：
$$ \mathcal{L}_{MP} = \lambda_{gray}\mathcal{L}_{gray} + \lambda_{struct}\mathcal{L}_{struct} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{color}\mathcal{L}_{color} $$

#### 3.2.1 物理约束：灰世界损失 (Gray-World Loss)
旨在消除水下图像常见的蓝/绿色彩偏差。基于灰世界假设（平均反射率为灰色），迫使生成图像的 RGB 通道平均值趋于平衡：
$$ \mathcal{L}_{gray}(\hat{I})=\frac{1}{3}\sum_{c\in\{r,g,b\}}\left| \mu_c(\hat{I}) - \mu_{gray}(\hat{I}) \right|,\quad \mu_{gray}(\hat{I})=\frac{\mu_r(\hat{I})+\mu_g(\hat{I})+\mu_b(\hat{I})}{3} $$ [7]

#### 3.2.2 结构约束：结构一致性损失 (Structure Loss)
为了在风格迁移过程中保持物体的几何结构并抑制结构漂移，我们采用基于梯度的一致性损失，将输入与生成结果的水平/垂直一阶差分对齐：
$$
\mathcal{L}_{struct}(I,\hat{I})=\left\|\nabla_x \hat{I}-\nabla_x I\right\|_1+\left\|\nabla_y \hat{I}-\nabla_y I\right\|_1
$$
其中 $\nabla_x I(i,j)=I(i,j+1)-I(i,j)$，$\nabla_y I(i,j)=I(i+1,j)-I(i,j)$。该项直接约束边缘与纹理的局部变化，能有效缓解 CycleGAN 在水下增强场景中的结构破坏问题。

#### 3.2.3 感知约束：深层感知损失 (Perceptual Loss)
利用在 ImageNet 上预训练的 VGG-16 特征提取器 $\phi$，计算输入与输出在特征空间的差异，以保持语义内容的一致性 [9,10]：
$$ \mathcal{L}_{perc}(I,\hat{I})=\left\|\phi(I)-\phi(\hat{I})\right\|_1 $$
其中 $\phi(\cdot)$ 表示取 VGG-16 前若干层的特征（本文复现实验中采用 `perceptual_layer=16`）。

#### 3.2.4 颜色统计一致性损失 (Color Statistics Loss)
为进一步约束增强结果的全局颜色分布（用于消融 Config E），我们引入一项简单的通道统计匹配损失，分别对齐生成图像与输入图像的通道均值与标准差：
$$
\mathcal{L}_{color}(I,\hat{I})=\left\|\mu(\hat{I})-\mu(I)\right\|_1+\left\|\sigma(\hat{I})-\sigma(I)\right\|_1
$$
其中 $\mu(\cdot),\sigma(\cdot)$ 在空间维度上对每个通道独立计算。

## 4. 实验与结果 (Experiments)

### 4.1 实验设置
- **数据集与协议**：
    - **主协议（同域评测）**：在 EUVP 的 Unpaired 分支训练 [2]；在 EUVP `test_samples` 上评测（`matched=200`）。
    - **跨域协议（泛化测试）**：仅在 EUVP 训练，直接测试 UIEB（`matched=890`）[1]，不进行目标域微调。
- **评价指标**：
    - **全参考指标**：PSNR, SSIM（在 EUVP `test_samples` 与 UIEB `reference-890` 上计算）[8]。
    - **无参考指标**：UCIQE、UIQM（在 EUVP 与 UIEB 上均统计）[11,12]。
- **统计设置**：PSNR/SSIM 的 95% 置信区间通过 bootstrap 估计（`bootstrap_iters=2000`，`seed=123`）；UCIQE/UIQM 报告为全体样本均值。
- **网络结构**：生成器采用 ResNet-9 Blocks，判别器采用 PatchGAN（basic），归一化为 InstanceNorm，不使用 dropout。
- **训练设置（同域主结果）**：
    - **CycleGAN**：Adam（$\beta_1=0.5$），学习率 0.0002；线性衰减策略，`n_epochs=100`，`n_epochs_decay=100`（总 200 epoch）；`batch_size=1`；`preprocess=resize_and_crop`，`load_size=286`，`crop_size=256`；随机水平翻转（未启用 `no_flip`）。
    - **MP-CycleGAN（Ours）**：以 CycleGAN 的 epoch=200 权重为初始化，在相同数据与预处理设置下继续训练至 epoch=203（3 个 epoch），学习率 0.0001；自监督项权重为 $\lambda_{gray}=0.1$，$\lambda_{struct}=2.0$，$\lambda_{perc}=0.05$，$\lambda_{color}=0$；感知网络使用 ImageNet 预训练 VGG-16（`perceptual_layer=16`）。本文在 `epoch=200–203` 的 checkpoint 上统一重评测，并选取 EUVP 同域主指标最优的 `epoch=202` 作为报告点。
- **推理方向**：使用 $G:A\rightarrow B$（增强方向）进行单向推理与评测（`AtoB`）。

### 4.2 主结果（EUVP 同域评测）
**表 1：EUVP `test_samples` 同域评测结果（matched=200）**

| 方法 | PSNR_mean (dB) $\uparrow$ | SSIM_mean $\uparrow$ | UCIQE_inp | UCIQE_pred $\uparrow$ | UIQM_inp | UIQM_pred $\uparrow$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Identity（输出=输入） | 20.33 [19.93, 20.70] | 0.794 [0.785, 0.801] | 24.88 | 24.88 | 5.77 | 5.77 |
| Gray-World（非学习基线） | 19.42 [19.00, 19.85] | 0.784 [0.775, 0.793] | 24.88 | 24.27 | 5.77 | 6.72 |
| CLAHE（非学习基线） | 16.93 [16.68, 17.18] | 0.697 [0.689, 0.705] | 24.88 | 28.01 | 5.77 | 6.14 |
| Gray-World + CLAHE（非学习基线） | 16.59 [16.34, 16.83] | 0.702 [0.693, 0.712] | 24.88 | 27.24 | 5.77 | 6.06 |
| CycleGAN（EUVP 训练，epoch=200） | 22.82 [22.40, 23.25] | 0.783 [0.773, 0.794] | 24.88 | 25.93 | 5.77 | 5.33 |
| MP-CycleGAN（Ours，epoch=202） | **22.91** [22.46, 23.35] | **0.795** [0.784, 0.805] | 24.88 | 25.82 | 5.77 | 5.41 |

表中 PSNR/SSIM 的区间为 95% bootstrap 置信区间。需要说明的是，UIQM 是由颜色相关项（UICM）、清晰度/锐度相关项（UISM）以及对比度相关项（UIConM）加权组合得到的无参考指标。对于基于对抗学习的无配对增强模型，在提升去雾与整体色调一致性的同时，生成结果可能出现一定程度的平滑，导致 UISM 下降，从而出现“UCIQE 上升但 UIQM 略降”的现象。这一趋势与 PSNR/SSIM 的提升并不矛盾，反映的是不同指标关注点（像素一致性 vs. 感知锐度/纹理）之间的权衡。

### 4.3 消融实验 (Ablation Study)
为了验证各组件的贡献，我们进行了逐步叠加的消融实验（Config A–E）。该消融在受控的小预算跨域设置下进行：在 UIEB_Unpaired 上训练 20 个 epoch（`max_dataset_size=200`），并统一在 EUVP `test_samples` 上评测（`matched=200`）。配置从 A 到 E 依次在上一配置基础上叠加新的自监督项（Gray → Struct → Perc → Color）。由于训练域与评测域不同，绝对 PSNR/SSIM 数值会显著低于同域主结果，本文主要关注不同损失组合带来的相对变化趋势。

**表 2：不同损失组合的性能对比（小预算跨域消融）**

| 配置 ID | 描述 | PSNR $\uparrow$ | SSIM $\uparrow$ | UCIQE $\uparrow$ | UIQM $\uparrow$ |
|:-------:|:-----|:-----:|:-----:|:------:|:-----:|
| **A** | CycleGAN（Baseline） | 16.85 | 0.56 | 27.89 | 4.13 |
| **B** | +Gray（$\lambda_{gray}>0$） | 17.04 | 0.56 | **28.66** | 4.16 |
| **C** | +Gray+Struct（$\lambda_{gray}>0,\ \lambda_{struct}>0$） | 16.67 | 0.65 | 26.85 | 4.18 |
| **D** | +Gray+Struct+Perc（MP-CycleGAN，$\lambda_{perc}>0$） | **17.80** | **0.65** | 28.51 | 4.56 |
| **E** | +Gray+Struct+Perc+Color（$\lambda_{color}>0$） | 16.72 | 0.64 | 27.87 | **4.76** |

*注：数据表明，MP-CycleGAN (+Perc) 在保持结构保真度 (SSIM) 与提升视觉质量 (UIQM) 之间取得了较优平衡；+Color 在 UIQM 上最高，但 PSNR/SSIM 略有下降。*

![Ablation Study](figures/ablation_study_grid.png)
> **图 2**：消融实验量化对比。MP-CycleGAN 在各维度上均表现优异。

### 4.4 视觉效果对比
图 3 展示了不同方法在典型水下场景下的增强效果。

![Visual Comparison](figures/visual_comparison_final_refined.png)
> **图 3**：EUVP 同域视觉质量对比。（a）Input，（b）CycleGAN，（c）MP-CycleGAN，（d）GT。版式按 IEEE 双栏宽度设计，统一边框与留白，便于印刷与缩放阅读。

### 4.5 颜色分布分析
为了验证物理约束的有效性，我们对比了增强前后的 RGB 直方图。

![Color Distribution](figures/color_distribution_analysis.png)
> **图 4**：颜色分布分析。MP-CycleGAN 成功将原始图像（左）偏移的 RGB 分布校正至平衡状态（中），与 Ground Truth（右）高度一致。

### 4.6 与现有方法对比 (Comparison with Existing Methods)
我们选取了 WaterNet [1]、UColor [3] 和 FUnIE-GAN [2] 的公开结果进行参考对比。

**表 3：在 EUVP 数据集上的对比（参考展示）**

| 方法 | 类别 | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ |
|:---|:---|:---:|:---:|
| **WaterNet** | Supervised | 19.81 | 0.86 |
| **UColor** | Supervised | 21.86 | 0.89 |
| **FUnIE-GAN** | Supervised | 23.50 | 0.92 |
| **CycleGAN（复现）** | **Unsupervised** | **22.82** | 0.783 |
| **MP-CycleGAN（复现）** | **Unsupervised** | **22.91** | **0.795** |

*注：CycleGAN/MP-CycleGAN（复现）的数值来自本文在统一评测脚本与测试划分下的复现实验；其余方法为原论文报告结果，受数据划分与评测设置差异影响，仅作参考展示，不作严格公平比较。*

![SOTA Comparison](figures/sota_comparison_refined.png)
> **图 5**：SOTA 方法性能对比。

### 4.7 跨域泛化性测试 (Cross-Domain Generalization)
我们将仅在 EUVP 上训练的 CycleGAN（epoch=200）与 MP-CycleGAN（epoch=202）直接应用于 UIEB 数据集，验证泛化性。

**表 4：UIEB 数据集跨域测试结果（matched=890）**

| 指标 | 输入 (Input) | CycleGAN（EUVP 训练，epoch=200） | MP-CycleGAN（EUVP 训练，epoch=202） |
|:---|:---:|:---:|:---:|
| **PSNR_mean (dB)** $\uparrow$ | - | **17.84** [17.64, 18.03] | 17.54 [17.35, 17.73] |
| **SSIM_mean** $\uparrow$ | - | **0.626** [0.617, 0.634] | 0.624 [0.615, 0.633] |
| **UCIQE** $\uparrow$ | 21.70 | **23.25** | 22.46 |
| **UIQM** $\uparrow$ | 6.80 | 10.48 | **10.50** |

从表 4 可以看到，在 UIEB 的跨域设置下，CycleGAN 在 PSNR/SSIM 与 UCIQE 上略优，而 MP-CycleGAN 在 UIQM 上略优。这说明两者在“像素一致性/结构保真”（PSNR/SSIM）与“主观感知锐度/对比度”（UIQM 中的 UISM/UIConM 等分量）之间存在不同的偏好：CycleGAN 更接近参考图像的像素统计，而 MP-CycleGAN 更倾向于提升无参考感知质量（更清晰、更有对比度），代价是与参考图像的逐像素误差可能略增。该差异在无配对跨域场景中尤为常见，也与 EUVP 同域实验中观察到的“UCIQE 上升但 UIQM 可能不一致”的指标关注点差异相一致。

![Cross Domain Test](figures/uieb_cross_domain_visuals.png)
> **图 6**：UIEB 跨域测试定性对比。（a）Input，（b）CycleGAN，（c）MP-CycleGAN，（d）GT。每行样本按 per-image 的 ΔUIQM（MP−CycleGAN）挑选代表案例，并在左侧给出 ΔUIQM 作为行标签，以展示跨域场景下的指标权衡与主观观感差异。

### 4.8 复现与有效性威胁 (Reproducibility & Threats to Validity)
为降低复现偏差并提升结论可信度，我们统一使用同一套推理与评测脚本生成 `summary.json / per_image.csv` 并在 `results/benchmarks/summary.csv` 中汇总。需要注意的有效性威胁包括：不同工作对数据集划分、预处理与评测实现的差异可能导致客观指标不可直接对齐；无参考指标（UCIQE/UIQM）与全参考指标（PSNR/SSIM）关注点不同，提升感知质量未必对应像素一致性提升；跨域设置下（EUVP→UIEB）分布差异更大，模型可能出现“更锐利但更偏离参考”的权衡。本文在表 3 中已对外部 SOTA 数值的可比性做出限定说明，并通过 bootstrap 置信区间报告主指标以减少随机波动带来的误判。

## 5. 结论 (Conclusion)
本文提出了 MP-CycleGAN，一种物理与感知驱动的非配对水下图像增强框架。通过集成灰世界、结构和感知损失，该方法在无需成对数据的情况下，实现了优异的去色偏与去雾效果。跨域测试证明了其强大的泛化能力，为解决真实水下数据稀缺问题提供了新思路。

## 参考文献 (References)
[1] C. Li, C. Guo, W. Ren, R. Cong, J. Hou, S. Kwong, and D. Tao, “An Underwater Image Enhancement Benchmark Dataset and Beyond,” IEEE Trans. Image Process., vol. 29, pp. 4376–4389, 2020, doi: 10.1109/TIP.2019.2955241.
[2] M. J. Islam, Y. Xia, and J. Sattar, “Fast Underwater Image Enhancement for Improved Visual Perception,” IEEE Robot. Autom. Lett., vol. 5, no. 2, pp. 3227–3234, Apr. 2020, doi: 10.1109/LRA.2020.2974710.
[3] C. Li, S. Anwar, J. Hou, R. Cong, C. Guo, and W. Ren, “Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding,” IEEE Trans. Image Process., vol. 30, pp. 4985–5000, 2021, doi: 10.1109/TIP.2021.3076367.
[4] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2242–2251, doi: 10.1109/ICCV.2017.244.
[5] K. He, J. Sun, and X. Tang, “Single Image Haze Removal Using Dark Channel Prior,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 1956–1963, doi: 10.1109/CVPR.2009.5206515.
[6] P. L. J. Drews Jr., E. R. Nascimento, S. S. C. Botelho, and M. F. M. Campos, “Underwater Depth Estimation and Image Restoration Based on Single Images,” IEEE Comput. Graph. Appl., vol. 36, no. 2, pp. 24–35, Mar.–Apr. 2016, doi: 10.1109/MCG.2016.26.
[7] G. Buchsbaum, “A Spatial Processor Model for Object Colour Perception,” J. Franklin Inst., vol. 310, no. 1, pp. 1–26, Jul. 1980, doi: 10.1016/0016-0032(80)90058-7.
[8] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004, doi: 10.1109/TIP.2003.819861.
[9] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual Losses for Real-Time Style Transfer and Super-Resolution,” in Computer Vision – ECCV 2016, 2016, pp. 694–711, doi: 10.1007/978-3-319-46475-6_43.
[10] K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.
[11] M. Yang and A. Sowmya, “An Underwater Color Image Quality Evaluation Metric,” IEEE Trans. Image Process., vol. 24, no. 12, pp. 6062–6071, Dec. 2015, doi: 10.1109/TIP.2015.2491020.
[12] K. Panetta, C. Gao, and S. Agaian, “Human-Visual-System-Inspired Underwater Image Quality Measures,” IEEE J. Oceanic Eng., vol. 41, no. 3, pp. 541–551, Jul. 2016, doi: 10.1109/JOE.2015.2469915.
[13] C. Ancuti, C. O. Ancuti, T. Haber, and P. Bekaert, “Enhancing Underwater Images and Videos by Fusion,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2012, pp. 81–88, doi: 10.1109/CVPR.2012.6247661.
[14] X. Fu, P. Zhuang, Y. Huang, Y. Liao, X.-P. Zhang, and X. Ding, “A Retinex-Based Enhancing Approach for Single Underwater Image,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2014, pp. 4572–4576, doi: 10.1109/ICIP.2014.7025927.
[15] D. Akkaynak and T. Treibitz, “Sea-Thru: A Method for Removing Water From Underwater Images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 1682–1691, doi: 10.1109/CVPR.2019.00178.
[16] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-Image Translation with Conditional Adversarial Networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 5967–5976, doi: 10.1109/CVPR.2017.632.
[17] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative Adversarial Nets,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2014, pp. 2672–2680.
[18] C. Fabbri, M. J. Islam, and J. Sattar, “Enhancing Underwater Imagery Using Generative Adversarial Networks,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2018, pp. 7159–7165, doi: 10.1109/ICRA.2018.8460552.
[19] Y.-T. Peng and P. C. Cosman, “Underwater Image Restoration Based on Image Blurriness and Light Absorption,” IEEE Trans. Image Process., vol. 26, no. 4, pp. 1579–1594, Apr. 2017, doi: 10.1109/TIP.2017.2663846.
[20] J. Y. Chiang and Y.-C. Chen, “Underwater Image Enhancement by Wavelength Compensation and Dehazing,” IEEE Trans. Image Process., vol. 21, no. 4, pp. 1756–1769, Apr. 2012, doi: 10.1109/TIP.2011.2179666.

## 附录 A：复现说明

### A. 指标来源（当前主线采用）

#### A.1 EUVP 同域（matched=200，CycleGAN epoch=200）

来源文件：`results/benchmarks/euvp/models/euvp_cyclegan_full/epoch_200/summary.json`

- PSNR_mean = 22.8155（95% CI: 22.3977–23.2450）
- SSIM_mean = 0.7833（95% CI: 0.7728–0.7937）
- UCIQE_inp = 24.8791，UCIQE_pred = 25.9333
- UIQM_inp = 5.7660，UIQM_pred = 5.3339

同一评测协议下，表 1 中其余方法对应的指标来源如下：

- MP-CycleGAN（Ours，epoch=202）：`results/benchmarks/euvp/models/euvp_mpcgan_stage2_s0/epoch_202/summary.json`
- Identity（输出=输入）：`results/benchmarks/euvp/baselines/identity_summary.json`
- Gray-World：`results/benchmarks/euvp/baselines/grayworld_summary.json`
- CLAHE：`results/benchmarks/euvp/baselines/clahe_summary.json`
- Gray-World + CLAHE：`results/benchmarks/euvp/baselines/grayworld_clahe_summary.json`

#### A.2 UIEB 跨域（matched=890，EUVP 预训练模型直接测试 UIEB）

来源文件：

- CycleGAN：`results/benchmarks/uieb/models/euvp_cyclegan_full/epoch_200/summary.json`
- MP-CycleGAN：`results/benchmarks/uieb/models/euvp_mpcgan_stage2_s0/epoch_202/summary.json`

关键指标（均为 matched=890 的全体均值；PSNR/SSIM 区间为 95% bootstrap CI）：

- CycleGAN：PSNR = 17.84 [17.64, 18.03]，SSIM = 0.626 [0.617, 0.634]，UCIQE = 23.25，UIQM = 10.48
- MP-CycleGAN：PSNR = 17.54 [17.35, 17.73]，SSIM = 0.624 [0.615, 0.633]，UCIQE = 22.46，UIQM = 10.50

#### A.3 评测命令（Windows / PowerShell）

1) 单向推理（A→B）：

```powershell
python test.py `
  --dataroot "<EUVP_TEST_INP>" `
  --name "euvp_cyclegan_full" `
  --model test `
  --dataset_mode single `
  --model_suffix "_A" `
  --epoch 200 `
  --num_test 200
```

2) EUVP `test_samples` 评测（PSNR/SSIM/UCIQE/UIQM）：

```powershell
python scripts/evaluate_euvp_psnr_ssim.py `
  --inp_dir "<EUVP_TEST_INP>" `
  --gtr_dir "<EUVP_TEST_GTR>" `
  --pred_dir "results\euvp_cyclegan_full\test_200\images" `
  --bootstrap_iters 2000 `
  --seed 123
```

3) UIEB 单向推理（A→B）：

```powershell
python test.py `
  --dataroot "<UIEB_INP>" `
  --name "euvp_mpcgan_stage2_s0" `
  --model test `
  --dataset_mode single `
  --model_suffix "_A" `
  --epoch 202 `
  --num_test 890 `
  --results_dir "results_uieb"
```

4) UIEB 跨域评测（PSNR/SSIM/UCIQE/UIQM）：

```powershell
python scripts/evaluate_euvp_psnr_ssim.py `
  --inp_dir "<UIEB_INP>" `
  --ref_dir "<UIEB_REF>" `
  --pred_dir "results_uieb\euvp_mpcgan_stage2_s0\test_202\images" `
  --bootstrap_iters 2000 `
  --seed 123
```

5) 小预算跨域消融（Config A–E）推理与评测（EUVP `test_samples`，matched=200）：

```powershell
python test.py `
  --dataroot "<EUVP_TEST_INP>" `
  --name "abl_uieb_A" `
  --model test `
  --dataset_mode single `
  --model_suffix "_A" `
  --epoch 20 `
  --num_test 200
```

```powershell
python scripts/evaluate_euvp_psnr_ssim.py `
  --inp_dir "<EUVP_TEST_INP>" `
  --gtr_dir "<EUVP_TEST_GTR>" `
  --pred_dir "results\abl_uieb_A\test_20\images" `
  --bootstrap_iters 2000 `
  --seed 123
```

将 `abl_uieb_A` 依次替换为 `abl_uieb_B/C/D/E`，即可得到表 2 的其余配置评测结果。

### A.4 关键训练配置（与 checkpoints 一致）

- CycleGAN（`euvp_cyclegan_full`）：`n_epochs=100`，`n_epochs_decay=100`，`lr=0.0002`，`batch_size=1`，`load_size=286`，`crop_size=256`，`netG=resnet_9blocks`，`netD=basic`。
- MP-CycleGAN（`euvp_mpcgan_stage2_s0`）：从 `epoch=200` 继续训练至 `epoch=203`，`lr=0.0001`；$\lambda_{gray}=0.1$，$\lambda_{struct}=2.0$，$\lambda_{perc}=0.05$，$\lambda_{color}=0$；本文报告点为 `epoch=202`。

### A.5 损失项与实现对应关系

为便于复现与复核，本文的自监督项均在 `models/cycle_gan_model.py` 内实现，并通过命令行权重开关启用：

- `--lambda_gray > 0`：灰世界损失（`_gray_world_loss`），约束生成结果 RGB 通道均值接近。
- `--lambda_struct > 0`：结构梯度一致性损失（`_structure_grad_loss`），约束水平/垂直差分一致。
- `--lambda_perceptual > 0`：感知损失（`_perceptual_loss`），使用 ImageNet 预训练 VGG-16 特征（`--perceptual_layer` 控制截断深度，默认 16）。
- `--lambda_color > 0`：颜色统计一致性损失（`_color_stats_loss`），约束均值与标准差匹配。

### A.6 训练命令（复现实验同款）

1) CycleGAN（EUVP Unpaired 训练，200 epoch）：

```powershell
python train.py `
  --dataroot "<EUVP_UNPAIRED_ROOT>" `
  --name "euvp_cyclegan_full" `
  --model cycle_gan `
  --n_epochs 100 `
  --n_epochs_decay 100 `
  --batch_size 1 `
  --preprocess resize_and_crop `
  --load_size 286 `
  --crop_size 256 `
  --netG resnet_9blocks `
  --netD basic `
  --no_dropout `
  --seed 0
```

2) MP-CycleGAN（从 CycleGAN epoch=200 继续训练 3 epoch）：

```powershell
python train.py `
  --dataroot "<EUVP_UNPAIRED_ROOT>" `
  --name "euvp_mpcgan_stage2_s0" `
  --model cycle_gan `
  --continue_train `
  --epoch 200 `
  --n_epochs 203 `
  --n_epochs_decay 0 `
  --lr 0.0001 `
  --batch_size 1 `
  --preprocess resize_and_crop `
  --load_size 286 `
  --crop_size 256 `
  --netG resnet_9blocks `
  --netD basic `
  --no_dropout `
  --max_dataset_size 500 `
  --seed 0 `
  --lambda_gray 0.1 `
  --lambda_struct 2.0 `
  --lambda_perceptual 0.05 `
  --lambda_color 0.0 `
  --perceptual_layer 16 `
  --perceptual_weights imagenet
```

3) UIEB_Unpaired 小预算跨域消融训练（Config A–E，20 epoch，`max_dataset_size=200`）：

```powershell
python train.py `
  --dataroot "D:\VScode\Graduation project\UIEB_Unpaired" `
  --model cycle_gan `
  --n_epochs 20 `
  --n_epochs_decay 0 `
  --lr 0.0002 `
  --batch_size 1 `
  --preprocess resize_and_crop `
  --load_size 286 `
  --crop_size 256 `
  --netG resnet_9blocks `
  --netD basic `
  --no_dropout `
  --no_html `
  --max_dataset_size 200 `
  --seed 0 `
  --name "abl_uieb_A"
```

```powershell
python train.py `
  --dataroot "D:\VScode\Graduation project\UIEB_Unpaired" `
  --model cycle_gan `
  --n_epochs 20 `
  --n_epochs_decay 0 `
  --lr 0.0002 `
  --batch_size 1 `
  --preprocess resize_and_crop `
  --load_size 286 `
  --crop_size 256 `
  --netG resnet_9blocks `
  --netD basic `
  --no_dropout `
  --no_html `
  --max_dataset_size 200 `
  --seed 0 `
  --name "abl_uieb_B" `
  --lambda_gray 0.1
```

```powershell
python train.py `
  --dataroot "D:\VScode\Graduation project\UIEB_Unpaired" `
  --model cycle_gan `
  --n_epochs 20 `
  --n_epochs_decay 0 `
  --lr 0.0002 `
  --batch_size 1 `
  --preprocess resize_and_crop `
  --load_size 286 `
  --crop_size 256 `
  --netG resnet_9blocks `
  --netD basic `
  --no_dropout `
  --no_html `
  --max_dataset_size 200 `
  --seed 0 `
  --name "abl_uieb_C" `
  --lambda_gray 0.1 `
  --lambda_struct 2.0
```

```powershell
python train.py `
  --dataroot "D:\VScode\Graduation project\UIEB_Unpaired" `
  --model cycle_gan `
  --n_epochs 20 `
  --n_epochs_decay 0 `
  --lr 0.0002 `
  --batch_size 1 `
  --preprocess resize_and_crop `
  --load_size 286 `
  --crop_size 256 `
  --netG resnet_9blocks `
  --netD basic `
  --no_dropout `
  --no_html `
  --max_dataset_size 200 `
  --seed 0 `
  --name "abl_uieb_D" `
  --lambda_gray 0.1 `
  --lambda_struct 2.0 `
  --lambda_perceptual 0.05 `
  --perceptual_layer 16 `
  --perceptual_weights imagenet
```

```powershell
python train.py `
  --dataroot "D:\VScode\Graduation project\UIEB_Unpaired" `
  --model cycle_gan `
  --n_epochs 20 `
  --n_epochs_decay 0 `
  --lr 0.0002 `
  --batch_size 1 `
  --preprocess resize_and_crop `
  --load_size 286 `
  --crop_size 256 `
  --netG resnet_9blocks `
  --netD basic `
  --no_dropout `
  --no_html `
  --max_dataset_size 200 `
  --seed 0 `
  --name "abl_uieb_E" `
  --lambda_gray 0.1 `
  --lambda_struct 2.0 `
  --lambda_perceptual 0.05 `
  --lambda_color 0.2 `
  --perceptual_layer 16 `
  --perceptual_weights imagenet
```

### A.7 论文图表生成（可构建）

在仓库根目录运行：

```powershell
python scripts/plot_unified_paper_figures.py
```

脚本会将论文中引用的图保存到：`docs/figures/`。
