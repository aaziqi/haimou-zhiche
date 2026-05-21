# Reproducibility Notes (MP-CycleGAN)

本说明用于配合你当前的 ICIP/ICASSP 初稿，明确“数据划分、训练设置、统一评测与统计置信区间”的复现信息。

## 1. 数据与评测集

- **EUVP（matched=200）**：项目内 `EUVP Dataset/test_samples/Inp` 与 `.../GTr`。
- **UIEB（matched=890）**：项目内 `UIEB/raw-890-s/raw-890/raw-890` 与 `UIEB/reference-890/reference-890`。

> 说明：论文中 UIEB 作为 **跨域测试集**（模型只在 EUVP 上训练，不在 UIEB 上微调）。

## 2. 训练设置（与你项目脚本一致）

- Backbone：CycleGAN 默认 **ResNet-9 generator + PatchGAN discriminator**，InstanceNorm。
- 输入尺寸：`256×256`（resize + random crop）。
- 优化器：Adam，`β1=0.5`。
- CycleGAN 训练：200 epochs，初始学习率 `2e-4`，后半程线性衰减。
- MP-CycleGAN：从 CycleGAN epoch 200 checkpoint 初始化，继续微调至 epoch 202（学习率 `1e-4`）。
- 关键损失权重（论文中已写入）：`λ_gray=0.1, λ_struct=2.0, λ_perc=0.05`。

## 3. 统一评测脚本（PSNR/SSIM + UCIQE/UIQM）

使用项目脚本：

- `pytorch-CycleGAN-and-pix2pix-master/scripts/evaluate_euvp_psnr_ssim.py`

该脚本对每张图像输出 per-image 指标（可选 CSV），并汇总：

- PSNR、SSIM：对匹配样本计算
- UCIQE、UIQM：对输入（Inp）与输出（Pred）均计算

### Bootstrap 置信区间（CI）

脚本参数（论文中已标注）：

- `bootstrap_iters=1000`
- `seed=123`

CI 计算实现位于 `_bootstrap_ci(...)`，输出为：

- `mean`
- `ci95_low`
- `ci95_high`

论文表格使用：`mean ± (ci95_high-ci95_low)/2`。

## 4. 参数量与推理速度

已用项目的网络定义统计：

- ResNet-9 generator 参数量：**11,378,179（约 11.38M）**
- 256×256，batch=1 推理：**16.99 ms/张（CUDA 设备）**

> 备注：推理耗时受硬件与环境影响很大，投稿时建议你在自己的实验环境（GPU型号）复测并在论文里注明硬件信息。

