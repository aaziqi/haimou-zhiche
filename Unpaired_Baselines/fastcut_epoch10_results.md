# FastCUT（epoch=10）里程碑结果

本文件记录 FastCUT 在 **epoch=10 checkpoint** 下的阶段性评测结果（用于“同类强基线”对比的里程碑检查）。

## 1) EUVP（in-domain，matched=200）
- PSNR：19.3230 dB（95% CI: [18.9485, 19.6614]）
- SSIM：0.7058（95% CI: [0.6952, 0.7158]）
- UCIQE：Inp 24.8791 → Pred 26.6266（GTr 26.2178）
- UIQM：Inp 5.7660 → Pred 5.0103（GTr 5.9009）

summary.json：
`pytorch-CycleGAN-and-pix2pix-master/results/benchmarks/euvp/external/FastCUT/epoch_10/summary.json`

## 2) UIEB（cross-domain，当前本地数据不完整，matched=841）
> 注意：你当前工程目录下的 UIEB 输入与参考图像数量不足 890（Inp=841，Ref=818），因此此处结果仅对应“当前可用子集”，不与论文主表（matched=890）严格等价。

- matched：841
- PSNR：17.8059 dB（95% CI: [17.6441, 17.9596]）
- SSIM：0.5856（95% CI: [0.5771, 0.5939]）
- UCIQE：Inp 21.3786 → Pred 26.4158（Ref 29.4727）
- UIQM：Inp 6.8919 → Pred 9.3234（Ref 6.3988）

summary.json：
`pytorch-CycleGAN-and-pix2pix-master/results/benchmarks/uieb/external/FastCUT/epoch_10/summary.json`

## 3) 下一步建议
1. 继续训练 FastCUT 到更高 epoch（例如 50/100/150/200），并在这些 checkpoint 上重复 EUVP/UIEB 评测，最终选择“最合理/最强”的 checkpoint 写入论文主表。
2. 若要与论文现有 UIEB 主结果（matched=890）严格可比，请补齐 UIEB 的 raw-890 与 reference-890 完整数据，再统一重跑对比方法与 FastCUT。

