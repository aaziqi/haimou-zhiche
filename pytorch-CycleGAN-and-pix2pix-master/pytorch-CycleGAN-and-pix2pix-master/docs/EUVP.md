# EUVP Unpaired Training, Testing, and Evaluation

- Dataset root (Windows): `d:\VScode\Graduation project\EUVP_Unpaired`
- Expected subfolders: `trainA`, `trainB`, optionally `validation` (single set for quick testing)

## Train CycleGAN

Use the provided PowerShell script:

```
# From repo root
./scripts/euvp_train_cyclegan.ps1 -DataRoot "d:\VScode\Graduation project\EUVP_Unpaired" -Name euvp_cyclegan -GpuIds 0
```

Notes:
- `--dataset_mode` defaults to `unaligned` for `--model cycle_gan`.
- Checkpoints save under `checkpoints/euvp_cyclegan`.

## Test on Single Folder (Validation or Inp)

If you only have one folder of images (e.g., `validation` or `test_samples/Inp`), use the single-direction test:

```
./scripts/euvp_test_single.ps1 `
  -DataRoot "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp" `
  -CheckpointsName "euvp_cyclegan" `
  -Epoch latest `
  -Direction AtoB
```

- For `test_samples/Inp`, set `-DataRoot` to that folder.
- Results saved under `results/<name>/test_<epoch>/` with images in `images/` and an HTML index.

## Optional: Prepare testA/testB from validation

If you want to run `--dataset_mode unaligned` in test phase, create `testA` by copying `validation`:

```
python scripts/prepare_euvp_unpaired.py --src "d:\VScode\Graduation project\EUVP_Unpaired" --dst "d:\VScode\Graduation project\EUVP_Unpaired" --make_test_from_validation
```

> Warning: `unaligned_dataset` expects both `testA` and `testB` to exist; if `testB` is empty, prefer `--model test` with `--dataset_mode single`.

## Evaluate PSNR/SSIM on EUVP test_samples

Assuming ground truths in `test_samples/GTr` and inputs in `test_samples/Inp`, and predictions saved by the test script:

```
python scripts/evaluate_euvp_psnr_ssim.py `
  --inp_dir "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp" `
  --gtr_dir "d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr" `
  --pred_dir "results\\euvp_cyclegan_full\\test_200\\images"
```

- Ensure OpenCV with contrib is installed for SSIM (`pip install opencv-contrib-python`).
- The evaluator matches files by basename. If names differ, adjust `--pred_dir` or rename outputs.

## Direction Clarification

- In CycleGAN, training learns both `A->B` (`netG_A`) and `B->A` (`netG_B`).
- The single test uses a chosen direction: set `-Direction AtoB` to enhance from domain A to B.

## Tips

- Use `--use_wandb` during training to log metrics and images to Weights & Biases.
- Control test epoch via `-Epoch` (e.g., `latest`, `50`, etc.).
- For large validation sets, increase `--num_test` in the test script.

## 实验结果汇总（EUVP 上的 CycleGAN 基线）

下表记录了 `euvp_cyclegan_full` 在 EUVP benchmark 上不同 epoch 的整体指标（`matched=200`），可作为后续改进方法的对比基线。

| Epoch | PSNR_mean | SSIM_mean | UCIQE_inp | UCIQE_pred | UIQM_inp | UIQM_pred | UCIQE_ref | UIQM_ref |
|------:|----------:|----------:|----------:|-----------:|---------:|----------:|----------:|---------:|
| 150   | 22.7348   | 0.7794    | 24.8791   | 25.3050    | 5.7660   | 5.4374    | 26.2178   | 5.9009   |
| 170   | 22.7684   | 0.7847    | 24.8791   | 26.1236    | 5.7660   | 5.4469    | 26.2178   | 5.9009   |
| 190   | 22.7560   | 0.7783    | 24.8791   | 25.7510    | 5.7660   | 5.3271    | 26.2178   | 5.9009   |
| 200   | 22.8155   | 0.7833    | 24.8791   | 25.9333    | 5.7660   | 5.3340    | 26.2178   | 5.9009   |

说明：
- 输入图像（Inp）的 UCIQE、UIQM 分别为 24.8791、5.7660；
- 参考图像（Ref）的 UCIQE、UIQM 分别为 26.2178、5.9009；
- epoch=200 的模型在 PSNR 上最高且 SSIM 接近最优，可作为 EUVP 上的主基线模型；若更偏向无参考指标（UCIQE/UIQM），epoch=170 表现更好。

## 实验结果汇总（UIEB 上的交叉测试）

使用在 EUVP 上训练得到的 `euvp_cyclegan_full`（epoch=200），直接在 UIEB 数据集上进行测试，得到如下指标（`matched=890`）：

| 模型                        | 数据集 | matched | UCIQE_inp | UCIQE_pred | UIQM_inp | UIQM_pred |
|-----------------------------|--------|--------:|----------:|-----------:|---------:|----------:|
| CycleGAN (EUVP 预训练 → 测试 UIEB) | UIEB   | 890     | 21.7036   | 23.2519    | 6.7984   | 10.4775   |

该结果作为“未在 UIEB 上进行目标域微调”时的交叉泛化基线，后续在 UIEB 上微调或引入额外自监督损失的模型，都可以与此行进行直接对比。

## 后续实验与论文表格规划（模板）

为便于后续论文撰写与结果整理，这里给出若干推荐表格结构（模板）。新实验完成后，直接在相应表格中新增行即可。

### 表 1：EUVP 上不同方法对比（主结果表）

| 方法                                           | PSNR_mean | SSIM_mean | UCIQE_inp | UCIQE_pred | UIQM_inp | UIQM_pred |
|----------------------------------------------|----------:|----------:|----------:|-----------:|---------:|----------:|
| Identity（不做增强，输出=输入）                    | 20.3279   | 0.7935    | 24.8791   | 24.8791    | 5.7660   | 5.7660    |
| Gray-World（非学习基线）                         | 19.4178   | 0.7838    | 24.8791   | 24.2706    | 5.7660   | 6.7160    |
| CLAHE（非学习基线）                              | 16.9289   | 0.6974    | 24.8791   | 28.0103    | 5.7660   | 6.1408    |
| Gray-World + CLAHE（非学习基线）                 | 16.5866   | 0.7023    | 24.8791   | 27.2447    | 5.7660   | 6.0583    |
| CycleGAN（EUVP 训练，euvp_cyclegan_full, e=200）  | 22.8155   | 0.7833    | 24.8791   | 25.9333    | 5.7660   | 5.3339    |
| CycleGAN + 自监督（跨域：UIEB 训练，配置 D, e=20） | 17.8048   | 0.6524    | 24.8791   | 28.5051    | 5.7660   | 4.5592    |
| CycleGAN + 自监督（跨域：UIEB 训练，配置 E, e=20） | 16.7193   | 0.6416    | 24.8791   | 27.8674    | 5.7660   | 4.7613    |

说明：
- 上表在 `EUVP Dataset/test_samples` 的前 200 张图像上评测（`matched=200`）；PSNR/SSIM 的 95% CI 以及 UIQM 分项（UICM/UISM/UIConM）汇总在 `results/benchmarks/summary.csv`。
- “跨域”代表训练域与评测域不同，仅用于观察泛化趋势；正式主结果建议统一在 EUVP 训练后再对比。

从 UIQM 的分项看，出现“UCIQE 上升但 UIQM 下降”的主要原因是 UISM（清晰度/锐度项）下降幅度较大：例如配置 D 在预测结果上 UICM 提升（2.8532→4.3170），但 UISM 从 16.0002 下降到 10.8299，导致总体 UIQM 低于输入；而 euvp_cyclegan_full 的 UISM 也从 16.0002 下降到 13.5886，因此 UIQM_pred 低于输入但幅度较小。后续若以 UIQM 为主目标，需要针对 UISM（如边缘/纹理保持、去噪与锐化平衡）单独做约束或训练策略优化。

### 表 2：UIEB 上预训练与微调对比

| 方法                                         | 训练策略                 | 数据集 | UCIQE_inp | UCIQE_pred | UIQM_inp | UIQM_pred |
|----------------------------------------------|--------------------------|--------|----------:|-----------:|---------:|----------:|
| CycleGAN（EUVP 预训练 → 直接测试 UIEB）      | 仅在 EUVP 上训练        | UIEB   | 21.7036   | 23.2519    | 6.7984   | 10.4775   |
| CycleGAN + 自监督（stage2_ss, 预训练 → 测试） | 仅在 EUVP 上训练        | UIEB   |           |            |          |           |
| CycleGAN + 自监督（EUVP 预训练 + UIEB 微调） | EUVP 预训练 + UIEB 微调 | UIEB   | 6.3918    | 6.7408     | 53.7683  | 55.2969   |

> 建议：在完成 UIEB 目标域微调实验后，将对应模型的 UCIQE/UIQM 指标填入本表，以展示迁移学习带来的提升。

### 表 3：损失组合消融实验（示例）

| 配置 ID | 使用灰世界 | 使用结构/梯度 | 使用感知 | 使用颜色统计 | PSNR_mean | SSIM_mean | UCIQE_pred | UIQM_pred |
|--------:|-----------:|--------------:|---------:|-------------:|----------:|----------:|-----------:|----------:|
| A       | 否         | 否            | 否       | 否           | 16.8475   | 0.5636    | 27.8875    | 4.1315    |
| B       | 是         | 否            | 否       | 否           | 17.0361   | 0.5615    | 28.6595    | 4.1623    |
| C       | 是         | 是            | 否       | 否           | 16.6692   | 0.6486    | 26.8521    | 4.1800    |
| D       | 是         | 是            | 是       | 否           | 17.8048   | 0.6524    | 28.5051    | 4.5592    |
| E       | 是         | 是            | 是       | 是           | 16.7193   | 0.6416    | 27.8674    | 4.7613    |

> 建议：选择 EUVP 或 UIEB 作为主评估数据集，在不同损失组合下训练模型，并将对应指标填入本表，用于论文中的消融实验分析。

本次消融在 `UIEB_Unpaired` 上训练：各配置训练 20 epoch（`max_dataset_size=200`），统一在 `EUVP Dataset/test_samples/Inp` 的前 200 张图像上以 `A→B` 方向推理，并使用 `GTr` 计算 PSNR/SSIM，同时统计预测结果的 `UCIQE/UIQM`（`matched=200`）。

从指标趋势看，灰世界项单独使用（B 相对 A）能显著提升 UCIQE，并带来小幅 PSNR 提升，说明其对颜色/亮度分布的约束在跨数据集推理时仍具一定稳健性。加入结构/梯度约束后（C），SSIM 明显提升但 UCIQE/PSNR 回落，表明结构项更偏向提升结构一致性，但在该小规模训练设置下可能引入风格偏移，导致像素级误差变大。

进一步引入感知损失（D）后，PSNR/SSIM 同时达到最优且 UCIQE 保持较高，体现出感知约束对跨域推理时的纹理稳定性更有帮助。加入颜色统计项（E）后，UIQM 有所提升但 PSNR/SSIM 与 UCIQE 均回落，说明颜色统计更强调全局一致性，可能在跨域场景中削弱了与参考图像的一致性。综合来看，D 在本实验设置下更均衡，可作为后续更大规模训练与同域/跨域评测的优先组合。

### 表 3b：损失组合消融实验（同域：EUVP 训练 → EUVP 测试）

| 配置 ID | 训练数据集 | epoch | seed | PSNR_mean | SSIM_mean | UCIQE_pred | UIQM_pred |
|--------:|-----------:|------:|-----:|----------:|----------:|-----------:|----------:|
| A       | EUVP_Unpaired | 201   | 0    | 22.7098   | 0.7776    | 25.9249    | 5.2959    |
| D       | EUVP_Unpaired | 50    | 0    | 21.9019   | 0.7716    | 25.2327    | 5.1254    |

说明：
- A 来自 `euvp_stage2_A_s0` 的 `latest`（≈epoch 201），训练时 `max_dataset_size=500`；
- D 来自 `euvp_abl_D_s0` 的 epoch=50（全量数据集设置）。

从同域结果看，配置 A 在较长训练后能同时取得更高的 PSNR/SSIM，并在 UCIQE/UIQM 上保持与基线接近的水平，说明“仅依靠对抗 + cycle + identity”的 CycleGAN 在 EUVP 上已具备较强的像素一致性优化能力；而加入自监督组合（D）在当前训练轮数下未能带来整体提升，更多体现为训练收敛速度与目标（PSNR/SSIM vs 无参考指标）之间的权衡。后续若希望自监督组合在同域上显著优于基线，建议统一训练预算（相同 max_dataset_size、epoch、seed），并对各损失权重做网格/贝叶斯搜索，或采用先基线预训练、再逐项启用自监督的渐进式策略。
