# Paper Preparation Guide: Underwater Image Enhancement via MP-CycleGAN (Multi-Perceptual CycleGAN)

## 1. Target Journal/Conference
- **Target**: ICIP.
- **Positioning**:
  - For ICIP, a clear problem framing + a simple but effective method with strong evidence is typically sufficient.
  - The key is not to claim “brand-new theory”, but to make the contribution clean and the experimental protocol airtight and reproducible.
- **Novelty & Evidence Checklist (ICIP-oriented)**:
  - A minimal and well-motivated modification over CycleGAN (multi-perceptual constraints) with ablation evidence for each term.
  - Fair comparisons: same datasets/splits and consistent evaluation scripts; avoid mixing “reported numbers” with “reproduced numbers” in the same main table.
  - Strong qualitative figure + 1–2 quantitative tables (main + ablation), plus cross-dataset generalization as a bonus.

### 1.1 Unified Experimental Protocol (Recommended for ICIP)

- **Main protocol (in-domain)**: Train on EUVP (unpaired) → test on EUVP `test_samples` (Inp/GTr/Ref) with matched-by-basename evaluation.
- **Generalization protocol (cross-dataset)**: Train on EUVP → test on UIEB without target-domain fine-tuning.
- **Ablation protocol**: Keep the same training budget (epochs, max_dataset_size, seed, preprocessing, direction) across all configs A–E.
- **Metrics**:
  - Full-reference: PSNR/SSIM on EUVP `test_samples` where GTr exists.
  - No-reference: UCIQE/UIQM on both EUVP and UIEB for additional evidence.
- **Baselines for ICIP main table**:
  - Identity (output=input).
  - 1–2 classic non-learning baselines (e.g., Gray-World, CLAHE).
  - CycleGAN baseline (same training budget as MP-CycleGAN).
  - Optional: 1 open-source underwater enhancement baseline you can actually run on your test split (only if reproducible under your protocol).

## 2. Abstract (Draft)
Underwater image enhancement is challenging due to wavelength-dependent light attenuation and scattering. Existing paired training methods suffer from a lack of large-scale ground truth data, while unpaired methods like CycleGAN often hallucinate artifacts or fail to preserve structure. We propose MP-CycleGAN (Multi-Perceptual CycleGAN), which integrates gray-world color constancy, structural similarity, and perceptual consistency constraints to guide unpaired translation. Extensive ablation studies on the EUVP and UIEB datasets demonstrate that the proposed constraint combination significantly improves visual quality and quantitative metrics (UCIQE/UIQM) compared to the baseline CycleGAN and other variants.

## 3. Methodology
### Framework
- **Baseline**: CycleGAN (Generator $G: A \to B$, $F: B \to A$, Discriminators $D_A, D_B$).
- **Novelty**: Integration of domain-specific multi-perceptual self-supervised constraints.

### Loss Functions (LaTeX)
1.  **Adversarial Loss**: $\mathcal{L}_{GAN}$
2.  **Cycle Consistency**: $\mathcal{L}_{cyc}$
3.  **Self-Supervised Losses**:
    - **Gray-World Loss** ($\mathcal{L}_{gray}$): Enforces average channel values to be neutral.
      $$ \mathcal{L}_{gray} = \sum_{c \in \{r,g,b\}} (mean(G(A)_c) - 0.5)^2 $$
    - **Structure Loss** ($\mathcal{L}_{struct}$): SSIM or Gradient-based loss to preserve high-frequency details.
    - **Perceptual Loss** ($\mathcal{L}_{perc}$): VGG-19 feature matching to preserve semantic content.
      $$ \mathcal{L}_{perc} = || \phi(A) - \phi(G(A)) ||_2^2 $$

## 4. Experiments & Results

### 4.1 Datasets
- **EUVP (Unpaired)**: Used for training and primary evaluation.
- **UIEB (Unpaired/Paired)**: Used for cross-domain generalization testing.

### 4.2 Implementation Details
- **Framework**: PyTorch CycleGAN.
- **Hardware**: NVIDIA GPU.
- **Hyperparameters**: Epochs=200 (Baseline), 50 (Ablation); Batch=1; LR=0.0002.

### 4.3 Ablation Study (Key Contribution)
We evaluated 5 configurations (A-E) to validate the contribution of each loss term.
- **Figure**: `docs/figures/ablation_uciqe.png`, `ablation_uiqm.png`, `ablation_psnr.png`
- **Observation**: 
  - **Config A (Baseline)**: Good PSNR but lower perceptual quality (lower UIQM).
  - **Config B (+Gray)**: Improves color balance (higher UCIQE) but can wash out colors.
  - **Config D (+Perc)**: Best trade-off, achieving high UIQM/UCIQE while maintaining structural fidelity (SSIM).

### 4.4 Visual Comparison
- **Figure**: `docs/figures/visual_comparison_sci.png`
- **Analysis**: The proposed method (Config D) removes green/blue casts more effectively than the baseline while preserving edge details better than Gray-world only variants.

### 4.5 Training Stability
- **Figure**: `docs/figures/training_loss_curves.png`
- **Analysis**: The addition of self-supervised losses does not destabilize training; convergence is similar to the baseline.

## 5. Gap Analysis (For ICIP Readiness)
1.  **Main-table fairness**: Ensure CycleGAN and MP-CycleGAN are trained under the same budget and evaluated by the same script/split.
2.  **Ablation rigor**: All configs A–E must share identical settings; otherwise move the low-budget ablation to appendix and clearly label it.
3.  **SOTA comparisons**: Only include methods you can run under the same protocol; otherwise label as “reported” and keep separate from reproduced comparisons.
4.  **Cross-dataset generalization**: Keep it as a strong supporting experiment; explicitly state “no fine-tuning on UIEB”.

## 6. Generated Figures Checklist
- [x] Training Loss Curves (`training_loss_curves.png`)
- [x] Visual Comparison Grid (`visual_comparison_sci.png`)
- [x] Ablation Metrics Bar Charts (`ablation_*.png`)
- [ ] Feature Map Visualization (Optional, adds technical depth)

## 7. Next Steps
1.  **Write**: Draft the sections based on the above outline.
2.  **Refine**: Replace "Baseline" with "CycleGAN" and "Proposed" with "MP-CycleGAN" in the text.
3.  **SOTA**: Find 1-2 external method results to add to Table 1 for comparison.

## 8. 详细写作包（可直接粘贴使用）

### 8.1 先统一论文命名与实验协议（强烈建议）

你当前材料中可能存在多种命名与多套实验数字（例如不同表格的 PSNR/SSIM 数量级差异）。在正式写作前，建议先做两件事：

- 命名统一：全文只使用 **MP-CycleGAN**（Multi-Perceptual CycleGAN）这一主名，其他称呼只在首次出现时作为别名说明一次（或不再出现）。
- 协议统一：正文只保留 1 套“主实验协议”，其余作为泛化/附录补充，并在每张表写清 `matched`、epoch、训练预算、是否同域/跨域。

推荐写法：

- 主协议 A（主结果表）：EUVP（unpaired）训练 → EUVP `test_samples` 评测（可算 PSNR/SSIM，同时报告 UCIQE/UIQM）
- 泛化协议 B（第二张表）：EUVP 训练 → UIEB 测试（强调跨域/未微调）
- 消融协议 C（消融表）：所有配置保持相同 epoch、max_dataset_size、seed、测试集与推理方向

### 8.2 标题（中英备选）

中文标题备选：

- 水下图像增强：一种融合物理约束与感知一致性的无配对 CycleGAN 方法
- MP-CycleGAN：基于多重感知约束的无配对水下图像增强
- 面向水下成像退化的多项自监督约束 CycleGAN 及其跨域泛化

英文标题备选：

- MP-CycleGAN: Unpaired Underwater Image Enhancement via Multi-Perceptual Constraints
- MP-CycleGAN: Multi-Perceptual CycleGAN for Unpaired Underwater Image Enhancement

### 8.3 摘要（中英）

#### 中文摘要（推荐）

水下图像由于波长相关吸收与散射效应，普遍存在明显的蓝绿色偏色、对比度下降与细节模糊等退化，严重影响水下视觉任务的可靠性。现有基于监督学习的增强方法通常依赖大规模成对数据进行训练，但在真实水下环境中获取像素级配对的“退化图像—清晰图像”样本成本高且难以覆盖多变场景，导致模型泛化受限。为此，本文提出一种无配对水下图像增强框架 MP-CycleGAN，在 CycleGAN 的双向循环对抗学习基础上，引入面向水下退化特性的多项自监督约束：采用灰世界颜色恒常性损失缓解全局色偏，使用结构一致性损失抑制几何结构漂移，并通过感知一致性损失增强语义与纹理保持能力。我们在 EUVP 与 UIEB 数据集上进行系统实验与消融分析，结果表明所提出的多项约束能够在不依赖配对数据的情况下显著提升增强结果的视觉质量与结构保真度，并在跨数据集测试中表现出更强的泛化能力。

关键词：水下图像增强；无配对学习；CycleGAN；自监督约束；灰世界；感知损失；结构一致性

#### 英文摘要（备选）

Underwater images often suffer from severe color casts, low contrast, and blurred details due to wavelength-dependent attenuation and scattering. Supervised enhancement methods rely on large-scale paired data, which is expensive and difficult to collect in real underwater environments, limiting generalization to diverse scenes. This paper presents MP-CycleGAN, an unpaired underwater image enhancement framework built upon CycleGAN with multi-perceptual self-supervised constraints tailored to underwater degradations. Specifically, we incorporate a gray-world color constancy loss to mitigate global color shifts, a structural consistency loss to preserve geometric structures, and a perceptual consistency loss to retain semantic and texture details. Extensive experiments and ablation studies on EUVP and UIEB demonstrate that the proposed constraints improve visual quality and structural fidelity without requiring paired supervision, and show stronger cross-dataset generalization.

### 8.4 引言（可直接粘贴）

#### 1.1 背景与问题

地球表面约 71% 被海洋覆盖，水下视觉在海洋资源勘探、水下机器人导航、海底工程检测与生态监测等任务中具有重要应用价值。然而，水下成像过程受到介质吸收与散射的显著影响：不同波长光的衰减速率不同导致颜色偏移（常见为蓝/绿偏），后向散射引起的雾化效应降低对比度与可见度，同时悬浮颗粒与噪声进一步破坏图像纹理细节。这些退化会显著影响后续检测、分割与识别等高层视觉任务的性能，因此水下图像增强通常被视为关键预处理环节。

#### 1.2 现有方法局限

传统基于物理先验的方法依赖特定成像假设，面对复杂多变的水下光照与水体类型时往往鲁棒性不足。近年来深度学习方法取得进展，其中监督学习策略通常需要“退化—清晰”成对数据以最小化像素级误差，但真实水下环境中获取严格配对样本极其困难：同一场景的“清晰图像”难以直接采集，合成数据与真实数据之间存在明显 domain gap，使得仅在合成或有限配对数据上训练的模型在真实场景中泛化能力受限。

无配对学习为缓解数据稀缺提供了可行途径。CycleGAN 等无配对图像翻译框架通过循环一致性约束在没有配对监督的条件下学习域间映射，但直接将 CycleGAN 用于水下增强仍可能出现结构漂移、纹理失真或伪影等问题。其根本原因在于对抗学习主要约束分布匹配，缺乏对“颜色校正—结构保持—语义一致”的任务特异约束。

#### 1.3 本文方法概述

为此，本文提出 MP-CycleGAN：在 CycleGAN 基础上引入面向水下退化的多重自监督约束。我们将水下增强目标拆解为三个互补的约束方向：颜色层面的全局色偏校正、结构层面的几何一致性保持、感知层面的语义与纹理一致性保持，并分别以灰世界损失、结构一致性损失（如 SSIM）与感知损失（如 VGG 特征一致性）实现。该组合在不引入额外配对监督的前提下，对生成结果施加可解释且与任务相匹配的约束，从而提升视觉质量与跨域泛化能力。

#### 1.4 贡献点（建议 3 条）

- 提出无配对水下图像增强框架 MP-CycleGAN，在 CycleGAN 上融合面向水下退化的多重感知自监督约束，实现对颜色、结构与感知一致性的联合建模。
- 给出系统消融实验与分析，验证灰世界约束、结构一致性约束与感知一致性约束对增强效果的独立贡献与互补性。
- 在 EUVP 与 UIEB 上进行实验评估，并通过跨数据集测试验证方法的泛化能力，同时提供可视化与统计分析支撑结论。

### 8.5 相关工作（写作结构建议）

建议按以下四类组织相关工作，每类 1 段总结 + 若干代表方法：

1. 基于物理模型与先验的方法：强调可解释性与泛化局限。
2. 基于监督学习的水下增强：强调配对数据成本与 domain gap。
3. 无配对图像翻译与水下增强：指出 CycleGAN 在增强任务上的结构漂移与伪影风险。
4. 自监督/结构一致性/感知一致性约束：解释 SSIM、perceptual loss 的作用，并过渡到本文的多约束组合。

### 8.6 方法（可直接粘贴的“审稿友好版”）

#### 3.1 问题定义

设水下退化域为 \(A\)，清晰/参考域为 \(B\)。给定无配对样本集合 \(\{x_i\}_{i=1}^{N_A}\subset A\) 与 \(\{y_j\}_{j=1}^{N_B}\subset B\)，目标是在不使用成对对应关系的前提下学习映射 \(G:A\rightarrow B\)，使得生成结果 \(\hat{y}=G(x)\) 在视觉质量与结构保真度上得到提升，同时避免结构漂移与伪影。

#### 3.2 CycleGAN 基线

生成器 \(G:A\rightarrow B\)、\(F:B\rightarrow A\)，判别器 \(D_B\) 与 \(D_A\)。总损失由对抗损失、循环一致性与 identity 约束组成，分别记为 \(\mathcal{L}_{GAN}\)、\(\mathcal{L}_{cyc}\)、\(\mathcal{L}_{idt}\)。

#### 3.3 多感知自监督约束（MP Losses）

定义总损失：

\[
\mathcal{L}_{total}=\mathcal{L}_{GAN}(G,D_B,A,B)+\mathcal{L}_{GAN}(F,D_A,B,A)+\lambda_{cyc}\mathcal{L}_{cyc}+\lambda_{idt}\mathcal{L}_{idt}+\mathcal{L}_{MP}
\]

其中

\[
\mathcal{L}_{MP}=\lambda_{gray}\mathcal{L}_{gray}+\lambda_{struct}\mathcal{L}_{struct}+\lambda_{perc}\mathcal{L}_{perc}
\]

灰世界损失（缓解色偏）：

\[
\mathcal{L}_{gray}(\hat{y})=\sum_{c\in\{r,g,b\}}(mean(\hat{y}_c)-\mu)^2
\]

结构一致性损失（抑制结构漂移，示例为 SSIM）：

\[
\mathcal{L}_{struct}(x,\hat{y})=1-SSIM(x,\hat{y})
\]

感知一致性损失（保持语义与纹理）：

\[
\mathcal{L}_{perc}(x,\hat{y})=\sum_{l\in\mathcal{S}}\|\phi_l(x)-\phi_l(\hat{y})\|_2^2
\]

### 8.7 实验（可复现写法清单）

建议在实验章节明确写清以下信息（缺一项都容易被审稿人追问）：

- 数据集：训练用 EUVP unpaired；评测用 EUVP `test_samples`（Inp/GTr/Ref）与 UIEB（说明是否 paired 子集）
- 推理方向：A→B（增强方向）
- 预处理：`--preprocess`、`load_size`、`crop_size` 与输入尺寸约束（例如 4 的倍数）
- 网络结构：`netG/netD`、norm、dropout
- 超参：lr、batch、epoch、衰减策略、\(\lambda\) 权重
- 指标：PSNR/SSIM（有参考子集）；UCIQE/UIQM（无参考）；同时报告可视化

### 8.8 表格与图（正文最强组合建议）

- 表 1（主结果，同域）：EUVP 训练 → EUVP `test_samples`，报告 PSNR/SSIM + UCIQE/UIQM，并写清 `matched`
- 表 2（消融）：A/B/C/D/E，确保训练预算一致，并列出每项 \(\lambda\)
- 表 3（跨域泛化）：EUVP 训练 → UIEB 测试（未微调），报告 UCIQE/UIQM（可选 paired 子集再补 PSNR/SSIM）
- 图 1：方法框架图（突出新增 MP losses）
- 图 2：消融柱状图/雷达图（突出最优配置）
- 图 3：视觉对比（含局部放大）
- 图 4：颜色分布分析（说明统计方式与样本数）

### 8.9 高风险点（正式投稿前必须处理）

- 同一篇论文中出现两套互相矛盾的数字：必须统一来源与协议，并在每张表中写清 matched/epoch/预算/方向。
- SOTA 对比公平性：若未在同一测试集复现对比方法，避免写“优于某某”这种结论，改写为“报告结果参考/不可直接比”或补齐复现实验。
- MOS 用户研究：若未实际实施，删除该节；若实施，补齐人数、样本数、随机化与双盲、统计方式。

### 8.10 结论段（可直接用）

本文提出 MP-CycleGAN，一种面向水下图像增强的无配对学习框架。不同于仅依赖对抗与循环一致性的通用图像翻译模型，我们进一步引入灰世界颜色恒常性、结构一致性与感知一致性三类自监督约束，从颜色校正、结构保持与语义/纹理一致性三个层面约束生成结果。实验与消融分析表明，多项约束能够有效缓解无配对训练中的结构漂移与伪影问题，在 EUVP 与 UIEB 上获得更好的视觉质量与更强的跨域泛化。未来工作将进一步探索与水下成像物理模型更紧密的可解释约束，并在更大规模、多水体类型数据上验证方法的稳健性与实用性。
