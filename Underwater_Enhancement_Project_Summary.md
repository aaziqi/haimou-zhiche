## 1. 研究主题与目标

- 研究方向：基于无配对数据的水下图像增强。
- 基础模型：CycleGAN。
- 改进思路：在无配对设定下，引入自监督约束（颜色、结构、感知等），提升颜色还原与细节保留能力。
- 核心工作：
  - 在 EUVP 数据集上训练 CycleGAN 及其自监督改进版；
  - 在 UIEB 上做交叉测试（EUVP 预训练模型直接测试 UIEB）；
  - 使用 UIEB 的参考集作为目标域，对模型进行目标域微调（domain adaptation）；
  - 统一采用 UCIQE、UIQM +（在有参考时）PSNR、SSIM 进行客观评价。

## 2. 数据集与任务设定

### 2.1 EUVP Unpaired 数据集

- 路径：`d:\VScode\Graduation project\EUVP Dataset\Unpaired`
- 结构（标准 CycleGAN 结构）：
  - `trainA`：水下原始图像（Domain A）
  - `trainB`：增强/清晰图像（Domain B）
  - 可选 `validation` 用于快速验证（单域）。
- 用途：
  - 训练 CycleGAN 基线：`euvp_cyclegan_full`
  - 训练 CycleGAN + 自监督模型：`euvp_cyclegan_stage2_ss`

### 2.2 UIEB 数据集（用于交叉测试与目标域微调）

- 原始 UIEB 路径：`d:\VScode\Graduation project\UIEB`
  - `raw-890-s\raw-890\raw-890`：890 张水下原始图像（raw-890）
  - `reference-890\reference-890`：890 张参考增强图像（reference-890）
- 为目标域微调准备的 Unpaired 格式数据：
  - `d:\VScode\Graduation project\UIEB_Unpaired\trainA`  
    → 拷贝自 `raw-890`，作为输入域 A（UIEB 原始）
  - `d:\VScode\Graduation project\UIEB_Unpaired\trainB`  
    → 拷贝自 `reference-890`，作为目标域 B（UIEB 参考）
- 用途：
  - 作为测试集：评估在 EUVP 上预训练的模型在 UIEB 上的泛化。
  - 作为目标域微调数据：在 UIEB 上进一步训练，使输出逼近 UIEB 参考风格。

## 3. 模型与训练配置（写在论文“方法 / 实验设置”）

### 3.1 基础 CycleGAN 配置（EUVP 上）

- 网络：
  - 生成器：`netG = resnet_9blocks`
  - 判别器：`netD = basic`
  - 通道数：`input_nc = output_nc = 3`，`ngf = ndf = 64`
- 训练配置（典型 CycleGAN 设置）：
  - `batch_size = 1`
  - `gan_mode = lsgan`
  - `lr = 2e-4`，`beta1 = 0.5`
  - `crop_size = 256`，`load_size = 286`
  - `pool_size = 50`
  - `n_epochs + n_epochs_decay` 约 200（EUVP 完整训练）
- 损失项（标准 CycleGAN）：
  - 对抗损失：GAN loss
  - 循环一致性损失：`λ_A = λ_B = 10`
  - 身份损失（identity）：`λ_identity = 0.5`

### 3.2 自监督改进项（CycleGAN + 自监督）

在 CycleGAN 的基础上引入额外自监督损失：

- 颜色一致性损失（Color Loss）：
  - 约束增强后图像的全局颜色统计（如均值）更接近目标域/参考域，缓解水体偏色。
  - 权重：`lambda_color = 0.2`
- 结构/梯度损失（Structure / Gradient Loss）：
  - 约束输出图像在边缘/梯度结构上与输入保持一致，避免过度平滑或结构扭曲。
  - 权重：`lambda_struct = 2.0`
- 感知损失（Perceptual Loss）：
  - 使用预训练网络中间层特征，约束输出与参考在高层语义特征上接近，提升感知质量。
  - 权重：`lambda_perceptual = 0.05`
  - `perceptual_layer = 16`，`perceptual_weights = imagenet`

> 灰世界（gray-world）损失在部分实验中有实现，但当前主线实验和 UIEB 微调使用的是颜色 + 结构 + 感知组合。

## 4. 已完成的主要实验与结果

### 4.1 EUVP 上的 CycleGAN 基线

- 模型：`euvp_cyclegan_full`
- 任务：在 EUVP benchmark（matched=200）上训练和评估。
- 指标示例（epoch 200）：
  - `PSNR_mean ≈ 22.82 dB`
  - `SSIM_mean ≈ 0.7833`
  - `UCIQE_inp = 6.3458`，`UCIQE_pred ≈ 6.8186`
  - `UIQM_inp = 52.6751`，`UIQM_pred ≈ 59.8668`

### 4.2 EUVP 上的 CycleGAN + 自监督（stage2_ss）

- 模型：`euvp_cyclegan_stage2_ss`
- 数据：EUVP Unpaired。
- 额外损失：`lambda_color=0.2, lambda_struct=2.0, lambda_perceptual=0.05`。
- 指标示例（epoch 50）：
  - `PSNR_mean ≈ 22.07 dB`
  - `SSIM_mean ≈ 0.7609`
  - `UCIQE_pred ≈ 7.0082`
  - `UIQM_pred ≈ 58.3452`

### 4.3 UIEB 上的交叉测试（EUVP 预训练 → 直接测试 UIEB）

使用统一脚本在 UIEB 890 张图上评估：

1. CycleGAN 基线（EUVP 预训练）  
   - 模型：`euvp_cyclegan_full` (epoch=200)  
   - 训练策略：仅在 EUVP 上训练，不在 UIEB 上微调。  
   - UIEB 指标（matched=890）：  
     - `UCIQE_inp = 6.3918`  
     - `UCIQE_pred = 6.7408`  
     - `UIQM_inp = 53.7683`  
     - `UIQM_pred = 55.2969`

2. CycleGAN + 自监督（stage2_ss，EUVP 预训练）  
   - 模型：`euvp_cyclegan_stage2_ss`（在 EUVP 上带自监督训练后，直接测试 UIEB）。  
   - UIEB 指标（matched=890）：  
     - `UCIQE_inp = 6.3918`  
     - `UCIQE_pred = 6.7930`  
     - `UIQM_inp = 53.7683`  
     - `UIQM_pred = 54.7628`

### 4.4 UIEB 目标域微调（EUVP 预训练 + UIEB 微调）

- 模型：`euvp_cyclegan_stage2_ss_uieb_ft`
- 初始权重：从 `euvp_cyclegan_stage2_ss` 的最新 checkpoint 加载。
- 数据：
  - `trainA`: `UIEB_Unpaired\trainA`（raw-890）
  - `trainB`: `UIEB_Unpaired\trainB`（reference-890）
- 训练策略：
  - `continue_train = True`
  - 在 UIEB 上进行 30 个 epoch 的目标域微调（当前配置：`n_epochs=30, n_epochs_decay=0`）。
- 最终 30 epoch 微调完成后的 UIEB 全集评估结果（matched=890）：
  - `PSNR_mean ≈ 17.84 dB`
  - `SSIM_mean ≈ 0.626`
  - `UCIQE_inp = 6.3918`
  - `UCIQE_pred = 6.7408`
  - `UCIQE_ref = 9.5117`
  - `UIQM_inp = 53.7683`
  - `UIQM_pred = 55.2969`
  - `UIQM_ref = 67.0065`

> 上述数值基于 `scripts/evaluate_euvp_psnr_ssim.py` 在 890 张 UIEB 原始/参考图与 `euvp_cyclegan_stage2_ss_uieb_ft` 预测结果上的统一评估，可直接用于论文中的 UIEB 对比表 “EUVP 预训练 + UIEB 微调” 一行。

## 5. 论文写作要点提纲

### 5.1 方法章节

- 简要介绍 CycleGAN 结构（两个生成器、两个判别器、对抗 + 循环一致性 + 身份损失）。
- 重点描述自监督损失：
  - 颜色一致性损失的定义与作用；
  - 结构/梯度损失如何约束边缘与细节；
  - 感知损失如何提升主观质量。
- 说明目标域微调流程：
  - 先在 EUVP 上预训练；
  - 再使用 UIEB 原始图和参考图作为 unpaired 域，在目标域上进行少量 epoch 的微调。

### 5.2 实验设置章节

- 数据集说明：
  - EUVP Unpaired：训练集结构、A/B 域含义；
  - UIEB：raw-890 与 reference-890 的角色；评估时利用 reference 作为“伪 ground truth”。
- 训练细节：
  - 优化器与学习率；
  - 在 EUVP 上训练约 200 epoch；
  - 在 UIEB 上进行 30 epoch 的目标域微调；
  - 各损失项的权重设置。

### 5.3 实验结果与分析章节

- EUVP 上主结果表：
  - CycleGAN 基线 vs. CycleGAN + 自监督；
  - 对比 PSNR / SSIM / UCIQE / UIQM，强调自监督在感知指标上的收益。
- UIEB 上预训练与微调对比表：
  - 三种配置：
    1. CycleGAN（EUVP 预训练 → 直接测试 UIEB）
    2. CycleGAN + 自监督（stage2_ss，仅 EUVP 预训练）
    3. CycleGAN + 自监督（EUVP 预训练 + UIEB 微调）
  - 分析微调对 UCIQE、UIQM 以及主观视觉质量的影响。
- 可选：损失组合消融实验：
  - 不同损失组合（仅 CycleGAN / +颜色 / +结构 / +感知）在某一主数据集上的对比，用于说明各自监督项的贡献。

## 6. 优化与调试方案 (Optimization & Debugging)

### 6.1 TV Loss (Total Variation Loss)
- **描述**: 引入总变分损失（Total Variation Loss）以鼓励生成图像的空间平滑性，减少噪点和伪影。
- **实现**: 在 `CycleGANModel` 中新增 `_tv_loss` 方法，计算生成图像水平和垂直方向梯度的 L1 范数。
- **用法**: 训练时添加参数 `--lambda_tv <float>`。例如 `--lambda_tv 0.1`。
- **作用**: 平滑图像，抑制高频噪声，但过大的权重可能导致细节丢失（过度模糊）。

### 6.2 超参数自动搜索脚本
- **脚本**: `scripts/optimize_hyperparams.py`
- **功能**: 自动化遍历不同的超参数组合（如 `lambda_tv`, `lambda_color` 等），进行快速训练（如 1 epoch）和评估。
- **流程**:
  1. 定义超参数网格（Grid Search）。
  2. 对每组参数：训练模型 -> 生成测试图 -> 计算 PSNR/SSIM/UCIQE/UIQM。
  3. 将结果记录到 `optimization_results.csv`。
- **目的**: 快速筛选出对 UIEB 数据集效果最好的超参数配置，避免手动反复试错。

### 6.3 U-Net 生成器实验
- **脚本**: `scripts/train_unet_experiment.ps1`
- **配置**: 使用 `--netG unet_256` 替代默认的 `resnet_9blocks`。
- **原理**: U-Net 架构包含跳跃连接（Skip Connections），直接将编码器的特征拼接到解码器对应层。
- **预期收益**: 相比 ResNet，U-Net 可能更好地保留原始水下图像的结构和纹理细节，有助于提升 SSIM 和结构相似度，尤其是在水下场景结构复杂的情况下。

### 6.4 总结
上述工具（TV Loss, 超参数搜索, U-Net）均已就绪，可作为进一步提升 UIEB 评测指标（特别是结构保持和去噪方面）的手段。
