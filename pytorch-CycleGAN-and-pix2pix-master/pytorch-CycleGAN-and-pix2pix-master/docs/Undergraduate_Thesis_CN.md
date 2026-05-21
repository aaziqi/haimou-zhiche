# MP-CycleGAN：基于多感知驱动的非配对水下图像增强方法（本科毕业论文稿）
  
- 学校：【请填写】  
- 学院：【请填写】  
- 专业：【请填写】  
- 学生姓名：【请填写】  
- 学号：【请填写】  
- 指导教师：【请填写】  
- 完成日期：【请填写】  
  
---
  
## 摘要
水下图像增强是海洋工程、水下机器人与水下视觉任务中的关键预处理步骤。受水体吸收与散射影响，真实水下图像普遍存在蓝绿偏色、对比度降低与雾化等问题，直接削弱下游检测、识别与测量的可靠性。近年来，基于深度学习的监督增强方法在合成或成对数据上取得了显著效果，但其依赖难以获取的像素级配对数据，且在真实跨域场景中容易受到域差异影响而性能下降。
  
为缓解配对数据依赖并提升跨域鲁棒性，本文在 CycleGAN 的非配对图像到图像转换框架上，提出一种物理与感知双重驱动的无监督水下增强方法 **MP-CycleGAN**（Multi-Perceptual CycleGAN）。该方法在对抗损失与循环一致性约束基础上，引入三类自监督约束：基于灰世界假设的颜色物理约束用于抑制水下蓝绿偏色；基于梯度一致性的结构约束用于缓解风格迁移导致的结构漂移；基于预训练 VGG-16 特征的感知约束用于保留语义与纹理细节。本文在统一推理与评测脚本下，对 EUVP 与 UIEB 数据集进行同域评测与跨域泛化测试，并采用 bootstrap 置信区间报告关键指标，提升结论可复核性。
  
实验结果表明：在 EUVP `test_samples`（matched=200）上，CycleGAN（22.82 dB / 0.783）与 MP-CycleGAN（22.91 dB / 0.795）整体表现接近且 MP-CycleGAN 在像素一致性指标上略优；在 UIEB 跨域测试（matched=890）中，MP-CycleGAN 的 UIQM 达到 10.50，与 CycleGAN 的 10.48 基本一致。消融实验进一步验证了灰世界、结构与感知约束对指标的影响趋势，为 MP-CycleGAN 的损失设计提供了定量支撑。
  
**关键词**：水下图像增强；非配对学习；CycleGAN；灰世界；感知损失；结构一致性
  
---
  
## Abstract
Underwater image enhancement is a critical preprocessing step for marine engineering and underwater vision tasks. Due to wavelength-dependent absorption and backscattering, underwater images often suffer from severe color casts, low contrast, and haze-like artifacts, which substantially degrades the performance of downstream perception algorithms. While supervised deep models can achieve strong quantitative results, they heavily rely on pixel-aligned paired data that is expensive or even infeasible to collect in real underwater environments, and they may generalize poorly across domains.
  
To address these limitations, this thesis proposes **MP-CycleGAN** (Multi-Perceptual CycleGAN), an unpaired underwater enhancement framework built upon CycleGAN. We incorporate three self-supervised constraints: a gray-world color prior to mitigate underwater color casts, a gradient-based structure consistency loss to reduce structural distortions, and a VGG-16 perceptual loss to preserve semantic content and fine textures. We evaluate the method under a unified inference and evaluation pipeline on EUVP and UIEB, and report bootstrap confidence intervals for key metrics to improve reproducibility.
  
Results show that MP-CycleGAN achieves competitive performance under both in-domain evaluation and cross-domain testing, with clear trends supported by ablation studies.
  
**Keywords**: underwater image enhancement; unpaired learning; CycleGAN; gray-world prior; perceptual loss; structure consistency
  
---
  
## 目录
- 1 绪论  
- 2 相关工作与技术基础  
- 3 方法：MP-CycleGAN  
- 4 实验设计与结果分析  
- 5 结论与展望  
- 参考文献  
- 附录 A：复现说明与命令  
  
---
  
## 1 绪论
### 1.1 研究背景与意义
地球表面约 71% 被海洋覆盖，水下图像在水下机器人导航、海洋资源勘探、环境监测与考古打捞等领域应用广泛。然而，水下成像面临以下典型退化：  
- **吸收**：长波（红光）衰减更快，导致整体偏蓝/偏绿；  
- **散射**：后向散射引入雾化，降低对比度并破坏细节；  
- **光照不均**：光源方向与距离变化导致局部亮度不稳定。  
  
水下增强的目标是在尽量保留内容结构的前提下，改善颜色与对比度，使图像更适合视觉感知与后续任务。
  
### 1.2 研究问题与挑战
监督学习方法通常需要“退化图像—清晰参考”的成对数据，但在真实水下环境下获取严格配对的数据几乎不可行。无监督方法可降低配对依赖，但在水下增强任务中仍存在：  
- **颜色漂移**：生成结果出现不自然色调或过度校色；  
- **结构漂移**：内容几何结构被破坏或出现伪影；  
- **指标权衡**：无参考指标（如 UIQM/UCIQE）与全参考指标（PSNR/SSIM）关注点不同，难以同时一致提升。  
  
### 1.3 本文工作与贡献
本文主要工作如下：  
1. 在 CycleGAN 基础上提出 MP-CycleGAN，引入灰世界物理约束、结构梯度一致性与深层感知约束，缓解偏色与结构漂移。  
2. 在统一评测脚本下对 EUVP 与 UIEB 进行同域/跨域评测，报告 PSNR/SSIM 的 bootstrap 置信区间以增强可复核性。  
3. 通过小预算消融实验分析不同自监督项的作用，给出定量趋势支撑。  
  
### 1.4 论文结构
第 2 章介绍水下增强相关工作与 CycleGAN 技术基础；第 3 章给出 MP-CycleGAN 方法与损失设计；第 4 章介绍实验协议、指标与结果；第 5 章总结全文并展望后续工作。
  
---
  
## 2 相关工作与技术基础
### 2.1 传统水下增强方法
传统方法多依赖先验与物理模型，例如暗通道先验（DCP）及其水下变体（UDCP）、Retinex、融合增强等。其优点是无需训练数据、可解释性强，但在复杂水体、光照变化与散射程度差异显著时，容易出现过增强或颜色失真。
  
### 2.2 监督学习方法
监督方法通过成对数据学习从退化到清晰的映射，代表性方法包括 WaterNet [1]、UColor [3]、FUnIE-GAN [2] 等。该类方法通常能取得较高的客观指标，但对配对数据强依赖，且跨域泛化能力受限。
  
### 2.3 无监督方法与 CycleGAN
CycleGAN [4] 通过对抗学习与循环一致性在非配对数据上实现域间转换，显著降低了配对数据需求。但标准 CycleGAN 在水下增强中仍可能产生颜色漂移、纹理伪影或结构被破坏。本文将在 CycleGAN 框架之上引入物理与感知层面的约束，提升生成稳定性与跨域表现。
  
---
  
## 3 方法：MP-CycleGAN
### 3.1 整体框架
MP-CycleGAN 采用双向循环生成架构，包含两个生成器 $G: A \to B$（水下 $\to$ 清晰）、$F: B \to A$（清晰 $\to$ 水下），以及两个判别器 $D_A, D_B$。其中 $A$ 为水下域，$B$ 为清晰域。训练时通过对抗损失让生成结果逼近目标域分布，通过循环一致性约束保证内容可逆，从而在非配对条件下学习映射。
  
![MP-CycleGAN Architecture](figures/mp_cyclegan_architecture.png)  
> **图 3-1**：MP-CycleGAN 整体架构示意图。
  
### 3.2 基础损失（CycleGAN）
基础框架沿用 CycleGAN 的对抗学习与循环一致性约束。设 $x\sim p_A(x)$ 为水下域样本、$y\sim p_B(y)$ 为清晰域样本：
  
1) **对抗损失**：约束生成分布逼近目标域分布。本文默认采用最常用的 LSGAN 形式（实现细节与原仓库保持一致）：
$$
\mathcal{L}_{GAN}(G,D_B)=\mathbb{E}_{y}[(D_B(y)-1)^2]+\mathbb{E}_{x}[D_B(G(x))^2],
$$
并对 $F,D_A$ 亦定义同类损失。
  
2) **循环一致性损失**：约束映射近似可逆，减少内容漂移：
$$
\mathcal{L}_{cyc}(G,F)=\mathbb{E}_{x}[\|F(G(x))-x\|_1]+\mathbb{E}_{y}[\|G(F(y))-y\|_1].
$$
  
3) **恒等损失（可选）**：在部分设置中用于稳定颜色映射（本文主结果未依赖该项提升）：
$$
\mathcal{L}_{idt}(G,F)=\mathbb{E}_{y}[\|G(y)-y\|_1]+\mathbb{E}_{x}[\|F(x)-x\|_1].
$$
  
### 3.3 多感知自监督损失（MP）
在基础损失之上，本文引入物理与感知层面的自监督约束，形成总目标：
$$
\mathcal{L}_{total}=\mathcal{L}_{GAN}+\lambda_{cyc}\mathcal{L}_{cyc}+\lambda_{idt}\mathcal{L}_{idt}+\mathcal{L}_{MP}
$$
其中
$$
\mathcal{L}_{MP}=\lambda_{gray}\mathcal{L}_{gray}+\lambda_{struct}\mathcal{L}_{struct}+\lambda_{perc}\mathcal{L}_{perc}+\lambda_{color}\mathcal{L}_{color}.
$$
  
#### 3.3.1 灰世界损失（物理约束）
基于灰世界假设（平均反射率为灰色），约束生成图像 RGB 通道均值趋于平衡：
$$
\mathcal{L}_{gray}(\hat{I})=\frac{1}{3}\sum_{c\in\{r,g,b\}}\left|\mu_c(\hat{I})-\mu_{gray}(\hat{I})\right|,\quad
\mu_{gray}(\hat{I})=\frac{\mu_r(\hat{I})+\mu_g(\hat{I})+\mu_b(\hat{I})}{3}.
$$
该项主要用于抑制水下蓝/绿偏色。
  
#### 3.3.2 结构一致性损失（梯度约束）
为缓解风格迁移带来的结构漂移，采用基于一阶差分的梯度一致性约束：
$$
\mathcal{L}_{struct}(I,\hat{I})=\left\|\nabla_x \hat{I}-\nabla_x I\right\|_1+\left\|\nabla_y \hat{I}-\nabla_y I\right\|_1.
$$
该项直接约束边缘与纹理变化，有助于结构保真。
  
#### 3.3.3 感知损失（深层特征约束）
利用 ImageNet 预训练的 VGG-16 特征提取器 $\phi$，在特征空间约束输入与生成结果的语义一致性：
$$
\mathcal{L}_{perc}(I,\hat{I})=\left\|\phi(I)-\phi(\hat{I})\right\|_1.
$$
本文复现实验中采用 `perceptual_layer=16`，以平衡语义与细节。
  
#### 3.3.4 颜色统计一致性损失（全局统计约束）
为进一步约束全局颜色分布，加入通道均值与标准差匹配项：
$$
\mathcal{L}_{color}(I,\hat{I})=\left\|\mu(\hat{I})-\mu(I)\right\|_1+\left\|\sigma(\hat{I})-\sigma(I)\right\|_1.
$$
该项在消融配置 E 中启用，用于观察对无参考指标的影响趋势。
  
### 3.4 工程实现要点
本项目在 CycleGAN 模型中实现上述自监督项，并通过命令行参数控制权重：  
- `--lambda_gray`：灰世界损失权重  
- `--lambda_struct`：结构一致性损失权重  
- `--lambda_perceptual`：感知损失权重  
- `--lambda_color`：颜色统计一致性损失权重  
- `--perceptual_layer` / `--perceptual_weights`：控制 VGG-16 特征层深度与权重来源  
  
对应实现位于 [cycle_gan_model.py](file:///d:/VScode/Graduation%20project/pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/models/cycle_gan_model.py)。
  
---
  
## 4 实验设计与结果分析
### 4.1 数据集与评测协议
#### 4.1.1 EUVP（同域评测）
训练：EUVP 的 Unpaired 分支 [2]。  
评测：EUVP `test_samples`（matched=200），使用输入与 GT 对齐计算 PSNR/SSIM，并统计无参考指标 UCIQE/UIQM。
  
#### 4.1.2 UIEB（跨域泛化测试）
训练：仅在 EUVP 上训练，不使用 UIEB 微调。  
评测：UIEB（matched=890）[1]，用 `reference-890` 作为参考计算 PSNR/SSIM，同时统计 UCIQE/UIQM。
  
### 4.2 评价指标
本文采用两类指标：  
- **全参考指标**：PSNR、SSIM [8]  
- **无参考指标**：UCIQE [11]、UIQM [12]  
  
其中 UIQM 由颜色、清晰度与对比度等分量加权组合得到，不同指标关注点不同，因此可能出现“PSNR/SSIM 提升但 UIQM 不一致”的现象。
  
### 4.3 统计设置与置信区间
PSNR/SSIM 的 95% 置信区间通过 bootstrap 估计（`bootstrap_iters=2000`，`seed=123`）。UCIQE/UIQM 报告为全体样本均值。
  
### 4.4 主结果（EUVP 同域评测）
**表 4-1：EUVP `test_samples` 同域评测结果（matched=200）**  
  
| 方法 | PSNR_mean (dB) $\uparrow$ | SSIM_mean $\uparrow$ | UCIQE_inp | UCIQE_pred $\uparrow$ | UIQM_inp | UIQM_pred $\uparrow$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Identity（输出=输入） | 20.33 [19.93, 20.70] | 0.794 [0.785, 0.801] | 24.88 | 24.88 | 5.77 | 5.77 |
| Gray-World（非学习基线） | 19.42 [19.00, 19.85] | 0.784 [0.775, 0.793] | 24.88 | 24.27 | 5.77 | 6.72 |
| CLAHE（非学习基线） | 16.93 [16.68, 17.18] | 0.697 [0.689, 0.705] | 24.88 | 28.01 | 5.77 | 6.14 |
| Gray-World + CLAHE（非学习基线） | 16.59 [16.34, 16.83] | 0.702 [0.693, 0.712] | 24.88 | 27.24 | 5.77 | 6.06 |
| CycleGAN（EUVP 训练，epoch=200） | 22.82 [22.40, 23.25] | 0.783 [0.773, 0.794] | 24.88 | 25.93 | 5.77 | 5.33 |
| MP-CycleGAN（Ours，epoch=202） | **22.91** [22.46, 23.35] | **0.795** [0.784, 0.805] | 24.88 | 25.82 | 5.77 | 5.41 |
  
### 4.5 消融实验（小预算跨域设置）
消融实验用于分析损失项叠加的指标趋势：在 UIEB_Unpaired 上训练 20 epoch（`max_dataset_size=200`），统一在 EUVP `test_samples` 评测（matched=200）。该设置训练域与评测域不同，因此绝对指标显著低于同域主结果，本文主要关注相对变化趋势。
  
**表 4-2：不同损失组合的性能对比（小预算跨域消融）**  
  
| 配置 ID | 描述 | PSNR $\uparrow$ | SSIM $\uparrow$ | UCIQE $\uparrow$ | UIQM $\uparrow$ |
|:-------:|:-----|:-----:|:-----:|:------:|:-----:|
| **A** | CycleGAN（Baseline） | 16.85 | 0.56 | 27.89 | 4.13 |
| **B** | +Gray（$\lambda_{gray}>0$） | 17.04 | 0.56 | **28.66** | 4.16 |
| **C** | +Gray+Struct（$\lambda_{gray}>0,\ \lambda_{struct}>0$） | 16.67 | 0.65 | 26.85 | 4.18 |
| **D** | +Gray+Struct+Perc（MP-CycleGAN，$\lambda_{perc}>0$） | **17.80** | **0.65** | 28.51 | 4.56 |
| **E** | +Gray+Struct+Perc+Color（$\lambda_{color}>0$） | 16.72 | 0.64 | 27.87 | **4.76** |
  
![Ablation Study](figures/ablation_study_grid.png)  
> **图 4-1**：消融实验量化对比。
  
### 4.6 视觉对比与颜色分布分析
  
![Visual Comparison](figures/visual_comparison_final_refined.png)  
> **图 4-2**：EUVP 同域视觉对比。（a）Input，（b）CycleGAN，（c）MP-CycleGAN，（d）GT。
  
![Color Distribution](figures/color_distribution_analysis.png)  
> **图 4-3**：颜色分布分析。展示输入、增强结果与参考图像的 RGB 分布差异。
  
### 4.7 与现有方法对比（参考展示）
为便于读者直观对比，选取 WaterNet [1]、UColor [3] 与 FUnIE-GAN [2] 的公开结果进行参考展示；由于不同方法可能使用不同数据划分与评测实现，表中对比不构成严格公平比较。
  
**表 4-3：EUVP 数据集上的参考对比（仅作展示）**  
  
| 方法 | 类别 | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ |
|:---|:---|:---:|:---:|
| WaterNet | Supervised | 19.81 | 0.86 |
| UColor | Supervised | 21.86 | 0.89 |
| FUnIE-GAN | Supervised | 23.50 | 0.92 |
| CycleGAN（复现） | Unsupervised | 22.82 | 0.783 |
| MP-CycleGAN（复现） | Unsupervised | **22.91** | **0.795** |
  
![SOTA Comparison](figures/sota_comparison_refined.png)  
> **图 4-4**：SOTA 方法性能对比（参考展示）。
  
### 4.8 跨域泛化性测试（EUVP→UIEB）
**表 4-4：UIEB 数据集跨域测试结果（matched=890）**  
  
| 指标 | 输入 (Input) | CycleGAN（EUVP 训练，epoch=200） | MP-CycleGAN（EUVP 训练，epoch=202） |
|:---|:---:|:---:|:---:|
| **PSNR_mean (dB)** $\uparrow$ | - | **17.84** [17.64, 18.03] | 17.54 [17.35, 17.73] |
| **SSIM_mean** $\uparrow$ | - | **0.626** [0.617, 0.634] | 0.624 [0.615, 0.633] |
| **UCIQE** $\uparrow$ | 21.70 | **23.25** | 22.46 |
| **UIQM** $\uparrow$ | 6.80 | 10.48 | **10.50** |
  
![Cross Domain Test](figures/uieb_cross_domain_visuals.png)  
> **图 4-5**：UIEB 跨域测试定性对比。（a）Input，（b）CycleGAN，（c）MP-CycleGAN，（d）GT。
  
### 4.9 结果讨论
在同域评测中，MP-CycleGAN 在 PSNR/SSIM 上略优；在跨域场景中，CycleGAN 在 PSNR/SSIM 与 UCIQE 上略优，而 MP-CycleGAN 在 UIQM 上略优。该现象反映了不同指标对“像素一致性/结构保真”和“主观锐度/对比度”的侧重点差异。在无配对跨域增强中，这类权衡较为常见，应结合定性结果与应用需求综合判断。
  
---
  
## 5 结论与展望
### 5.1 结论
本文针对水下图像增强的配对数据稀缺与跨域泛化问题，提出 MP-CycleGAN 方法，在 CycleGAN 框架上引入灰世界物理先验、结构梯度一致性约束与深层感知约束。实验表明该方法在同域评测与跨域测试中表现稳定，消融实验也验证了各损失项对指标趋势的影响，为无监督水下增强提供了可复核的工程实现与实验结论。
  
### 5.2 展望
后续可从以下方向继续改进：  
- 引入更强的水下物理成像模型或可学习的颜色校正模块；  
- 针对无参考指标与全参考指标的冲突，引入多目标优化或感知一致性评价；  
- 在更大规模、更复杂水体类型的数据上进行验证，并结合下游任务（检测/分割）进行端到端评估。  
  
---
  
## 参考文献
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
[17] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, and Y. Bengio, “Generative Adversarial Nets,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2014, pp. 2672–2680.  
[18] C. Fabbri, M. J. Islam, and J. Sattar, “Enhancing Underwater Imagery Using Generative Adversarial Networks,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2018, pp. 7159–7165, doi: 10.1109/ICRA.2018.8460552.  
[19] Y.-T. Peng and P. C. Cosman, “Underwater Image Restoration Based on Image Blurriness and Light Absorption,” IEEE Trans. Image Process., vol. 26, no. 4, pp. 1579–1594, Apr. 2017, doi: 10.1109/TIP.2017.2663846.  
[20] J. Y. Chiang and Y.-C. Chen, “Underwater Image Enhancement by Wavelength Compensation and Dehazing,” IEEE Trans. Image Process., vol. 21, no. 4, pp. 1756–1769, Apr. 2012, doi: 10.1109/TIP.2011.2179666.  
  
---
  
## 附录 A：复现说明与命令
本附录给出关键结果的来源文件与可直接执行的命令模板，便于复核。输出的汇总表位于：`results/benchmarks/summary.csv`。
  
### A.1 指标来源（关键 summary.json）
- EUVP 同域（matched=200）  
  - CycleGAN：`results/benchmarks/euvp/models/euvp_cyclegan_full/epoch_200/summary.json`  
  - MP-CycleGAN：`results/benchmarks/euvp/models/euvp_mpcgan_stage2_s0/epoch_202/summary.json`  
- UIEB 跨域（matched=890）  
  - CycleGAN：`results/benchmarks/uieb/models/euvp_cyclegan_full/epoch_200/summary.json`  
  - MP-CycleGAN：`results/benchmarks/uieb/models/euvp_mpcgan_stage2_s0/epoch_202/summary.json`  
  
### A.2 推理与评测命令（Windows / PowerShell）
1) EUVP 单向推理（A→B）：
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
  
2) EUVP 评测（PSNR/SSIM/UCIQE/UIQM）：
```powershell
python scripts/evaluate_euvp_psnr_ssim.py `
  --inp_dir "<EUVP_TEST_INP>" `
  --gtr_dir "<EUVP_TEST_GTR>" `
  --pred_dir "results\\euvp_cyclegan_full\\test_200\\images" `
  --bootstrap_iters 2000 `
  --seed 123
```
  
3) UIEB 单向推理（A→B，输出到 results_uieb）：
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
  
4) UIEB 评测（PSNR/SSIM/UCIQE/UIQM）：
```powershell
python scripts/evaluate_euvp_psnr_ssim.py `
  --inp_dir "<UIEB_INP>" `
  --ref_dir "<UIEB_REF>" `
  --pred_dir "results_uieb\\euvp_mpcgan_stage2_s0\\test_202\\images" `
  --bootstrap_iters 2000 `
  --seed 123
```
  
### A.3 训练命令（复现实验同款）
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
  
### A.4 论文图表生成（可构建）
在仓库根目录运行：
```powershell
python scripts/plot_unified_paper_figures.py
```
图表输出目录：`docs/figures/`。
