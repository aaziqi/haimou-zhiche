# 基于 CycleGAN 的水下图像增强实验报告（EUVP / Unpaired）

作者：XXX（请替换）  
日期：2025-12-19  

## 环境

- 操作系统：Windows
- Python：3.11.9
- PyTorch：2.5.1+cu121（CUDA 12.1，cuDNN 90100）
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU
- 代码仓库：`pytorch-CycleGAN-and-pix2pix-master`
- 训练数据：`d:\VScode\Graduation project\EUVP Dataset\Unpaired`
- 测试/评估数据：`d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp` / `GTr`

## 摘要（100–200 字）

针对水下图像普遍存在的颜色偏移、对比度低与雾化等问题，本文基于 EUVP 非配对数据集训练 CycleGAN，实现从退化域到增强域的单向图像增强。模型采用 ResNet 生成器与 PatchGAN 判别器，结合 LSGAN 对抗损失、循环一致性损失与身份映射损失进行优化，并在 EUVP `test_samples` 上用 PSNR、SSIM 及脚本实现的 UCIQE/UIQM 进行评估。实验表明，`latest_net_G_A.pth` 在 200 对样本上取得最高 PSNR/SSIM 与 UIQM，整体视觉质量与可用性优于较早 epoch。

## 引言（背景、目标与方法）

水下成像受光吸收与散射影响显著，往往出现偏蓝/偏绿、细节丢失与对比度不足，影响后续检测与识别任务。传统基于物理模型的方法对场景假设敏感，而深度学习方法可在数据驱动下学习复杂退化到清晰图像的映射。

本文目标是在**无成对标注**条件下进行水下图像增强。为此采用 CycleGAN，通过学习两个域之间的双向映射并引入循环一致性约束，实现对退化输入的增强输出。测试阶段使用单向生成器（A→B）对输入图像做推理，并对不同 epoch 的模型进行对比选择最优权重。

## 数据与预处理

### 数据集

- 训练：EUVP Unpaired（目录需包含 `trainA` 与 `trainB`），训练采用 `--dataset_mode unaligned`（见 `models/cycle_gan_model.py:12` 的说明）
- 测试/评估：EUVP `test_samples`
  - 输入：`...\test_samples\Inp`
  - 参考真值：`...\test_samples\GTr`

### 预处理与归一化

本项目的数据增强与预处理由 `data/base_dataset.py` 实现：
- `resize_and_crop`：先缩放到 `load_size`，再裁剪到 `crop_size`（`data/base_dataset.py:68-97`）
- 翻转：当 `--no_flip` 未启用时，可随机水平翻转（`data/base_dataset.py:101-106`）
- 张量与归一化：`ToTensor()` 后做 `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`，将像素归一化到约 \([-1,1]\)（`data/base_dataset.py:107-113`）

本次实验的推理配置中 `crop_size=256`、`load_size=256`、`preprocess=resize_and_crop`（见 `checkpoints/euvp_cyclegan_full/test_opt.txt:5,17,34`）。

## 方法实现细节（网络结构、损失函数、训练策略）

### 网络结构

- 生成器：默认使用 `resnet_9blocks`（`models/cycle_gan_model.py:13-15`），在 `models/networks.py` 中由 `define_G` 选择 `ResnetGenerator(..., n_blocks=9)`（`models/networks.py:132-160`）
- 判别器：默认 `basic` PatchGAN（`models/cycle_gan_model.py:13-15`；`models/networks.py:163-203`）

### 损失函数

CycleGAN 的关键超参与损失在 `models/cycle_gan_model.py` 中给出：
- 对抗损失：`GANLoss(opt.gan_mode)`，默认 `gan_mode=lsgan`（`models/cycle_gan_model.py:92`；`options/train_options.py:29`）
- 循环一致性：`L1Loss`，权重 `lambda_A=10`、`lambda_B=10`（`models/cycle_gan_model.py:42-44,93-94,179-181`）
- 身份映射：`lambda_identity=0.5`（`models/cycle_gan_model.py:45-49,163-169`）

整体生成器损失由对抗项、cycle 项与 identity 项加和得到（`models/cycle_gan_model.py:174-184`）。

### 训练策略

- 优化器：Adam，默认 `lr=2e-4`、`beta1=0.5`（`options/train_options.py:27-28`；`models/cycle_gan_model.py:96-97`）
- 学习率策略：默认 `lr_policy=linear`，前 `n_epochs=100` 保持不变，后 `n_epochs_decay=100` 线性衰减（`options/train_options.py:25-26,31`；`models/networks.py:51-63`）
- 图像缓冲池：`pool_size=50`（`options/train_options.py:30`；`models/cycle_gan_model.py:89-90`）
- 保存频率：默认每 `save_epoch_freq=5` 个 epoch 保存一次（`options/train_options.py:19`），与当前实验目录下的 `5/10/15/20/25` checkpoint 一致（`checkpoints/euvp_cyclegan_full/*.pth`）

## 实验设置（超参）

### 训练相关（默认值 + 本次配置）

以下为本项目默认训练超参（未显式修改时生效）：
- `batch_size=1`（与本次测试配置一致，`checkpoints/euvp_cyclegan_full/test_opt.txt:3`）
- `lr=0.0002`，`beta1=0.5`（`options/train_options.py:27-28`）
- `gan_mode=lsgan`（`options/train_options.py:29`）
- `lambda_A=10.0`，`lambda_B=10.0`，`lambda_identity=0.5`（`models/cycle_gan_model.py:42-49`）
- `netG=resnet_9blocks`，`netD=basic`，`norm=instance`（见 `checkpoints/euvp_cyclegan_full/test_opt.txt:24-29`）

### 训练进度

训练日志显示已运行到 `epoch: 27`（例如 `checkpoints/euvp_cyclegan_full/loss_log.txt:1668`）。

## 定量结果（表格与分析）

### 评估协议

- 数据：EUVP `test_samples` 中 200 对样本（脚本输出 `Matched pairs: 200`）
- 指标：
  - PSNR：OpenCV `cv2.PSNR`（`scripts/evaluate_euvp_psnr_ssim.py:23-24`）
  - SSIM：脚本内实现（`scripts/evaluate_euvp_psnr_ssim.py:44-73`）
  - UCIQE/UIQM：脚本中基于颜色空间统计的实现（`scripts/evaluate_euvp_psnr_ssim.py:27-41`）

说明：UCIQE/UIQM 在本文中严格以 `scripts/evaluate_euvp_psnr_ssim.py` 的实现为准，便于复现实验数值。

### 不同 epoch 对比

| 生成器权重 | PSNR↑ (dB) | SSIM↑ | UCIQE(Pred)↑ | UIQM(Pred)↑ |
|---|---:|---:|---:|---:|
| `5_net_G_A.pth` | 19.8761 | 0.6866 | 6.4329 | 53.0334 |
| `10_net_G_A.pth` | 20.7427 | 0.7284 | 7.6957 | 58.9168 |
| `15_net_G_A.pth` | 20.0407 | 0.7110 | 6.4604 | 56.1009 |
| `20_net_G_A.pth` | 20.9639 | 0.6801 | 6.9864 | 59.7708 |
| `25_net_G_A.pth` | 20.6443 | 0.7062 | 6.1701 | 57.5564 |
| `latest_net_G_A.pth` | **21.6290** | **0.7475** | 6.6984 | **62.0109** |

### 分析

- 综合 PSNR/SSIM，`latest` 明显优于 5/10/15/20/25，说明模型在训练中后期仍在提升对结构与细节的重建一致性。
- UIQM(Pred) 在 `latest` 达到最高（62.0109），对比输入的 UIQM(Inp)=52.6751（脚本输出），主观上更容易对应到清晰度与可辨识度的提升。
- `10` 的 UCIQE(Pred) 偏高，但其 PSNR/SSIM 与 UIQM 低于 `latest`；这通常意味着“色彩对比/饱和度”增强更强，但不一定带来更好的结构一致性与整体保真度。

## 定性结果（图像示例与讨论）

### 生成图像整理

- 最佳模型权重：`checkpoints/euvp_cyclegan_full/latest_net_G_A.pth`
- 推理输出目录：`results/euvp_cyclegan_full/test_latest/images`
  - `*_real.png`：模型输入（预处理后）
  - `*_fake.png`：模型输出（增强后）

### 示例展示

- 项目自带的推理可视化：`results/euvp_cyclegan_full/test_latest/index.html`
- 本次整理的“输入/输出/真值”对比页：`results/euvp_cyclegan_full/test_latest/best_examples.html`

### 现象讨论

在示例中常见改善包括：
- 偏色缓解：蓝绿偏色被部分校正，白平衡更接近参考域
- 对比度提升：暗部细节更可见，局部对比增强
- 纹理恢复：部分细节变清晰，但在极端雾化/严重散射场景仍可能出现过增强与伪影

## 结论与未来工作

本文在 EUVP 非配对数据上训练 CycleGAN，并在 `test_samples` 上完成多 epoch 对比。结果显示 `latest_net_G_A.pth` 在 PSNR/SSIM 与 UIQM 上最佳，可作为后续论文展示与下游任务预处理的默认模型。

后续工作方向：
- 引入更贴近水下成像的损失与约束（例如颜色恒常、Retinex 先验、频域一致性）
- 更完善的无参考指标与主观实验（用户打分、下游检测性能）
- 尝试更强的生成器结构（注意力、UNet/Transformer）与更稳健的训练策略（EMA、谱归一化等）

## 附录：关键代码片段与复现说明

### A.1 单向推理模型（TestModel）

`models/test_model.py` 中通过 `--model_suffix` 加载对应权重（`models/test_model.py:12-31,46-52`），例如 `_A` 对应 `latest_net_G_A.pth`。

### A.2 CycleGAN 损失定义

CycleGAN 的损失与权重详见 `models/cycle_gan_model.py:31-49,157-184`。

### A.3 复现命令（Windows / PowerShell）

1) 推理（A→B，使用最佳模型 `latest_net_G_A.pth`）：

```powershell
python test.py `
  --dataroot "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp" `
  --name "euvp_cyclegan_full" `
  --model test `
  --dataset_mode single `
  --model_suffix "_A" `
  --epoch latest `
  --num_test 200
```

2) 评估（PSNR/SSIM/UCIQE/UIQM）：

```powershell
python scripts/evaluate_euvp_psnr_ssim.py `
  --inp_dir "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp" `
  --gtr_dir "d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr" `
  --pred_dir "results\euvp_cyclegan_full\test_latest\images"
```

3) 多 epoch 对比（示例）：

```powershell
$epochs = @("5","10","15","20","25","latest")
foreach ($e in $epochs) {
  python scripts/evaluate_euvp_psnr_ssim.py `
    --inp_dir "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp" `
    --gtr_dir "d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr" `
    --pred_dir "results\euvp_cyclegan_full\test_$e\images"
}
```

