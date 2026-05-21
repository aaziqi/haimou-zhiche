# TraeIDE 实验工作汇总（CUT/FastCUT 对比补充）

本文档用于你在 TraeIDE 里接续运行与整理论文实验，包含：已完成工作、当前运行状态、监测方法、下一步任务与建议命令。

---

## 0. 统一约定（目录）

### 训练/权重
`Unpaired_Baselines/checkpoints/`

### 推理输出（图片结果）
`Unpaired_Baselines/results/`

### 统一评测脚本（与你论文现有结果同一口径）
`pytorch-CycleGAN-and-pix2pix-master/scripts/evaluate_euvp_psnr_ssim.py`

评测输出（summary.json）统一放到：
`pytorch-CycleGAN-and-pix2pix-master/results/benchmarks/<dataset>/external/<method>/epoch_<k>/summary.json`

---

## 1. 已完成的工作（截至当前）

### 1.1 FastCUT 训练
- 训练名：`euvp_fastcut_full`
- 训练集：`EUVP_Unpaired`
- 当前 checkpoint 已完整覆盖：`10, 20, ..., 200`（见 `*_net_G.pth`）

权重目录：
`Unpaired_Baselines/checkpoints/euvp_fastcut_full/`

### 1.2 FastCUT（epoch=10）里程碑评测（已完成）

**EUVP matched=200（in-domain）**：
- 评测 summary：
`pytorch-CycleGAN-and-pix2pix-master/results/benchmarks/euvp/external/FastCUT/epoch_10/summary.json`

**UIEB（cross-domain）**：
- 评测 summary：
`pytorch-CycleGAN-and-pix2pix-master/results/benchmarks/uieb/external/FastCUT/epoch_10/summary.json`

> 注意：你当前工程目录下 UIEB 数据不完整（Inp/Ref 数量不足 890），因此 UIEB 的 PSNR/SSIM 对比只能在“当前可用子集”上成立；若要与论文 matched=890 严格一致，需要补齐 raw-890 与 reference-890 完整数据。

里程碑结果汇总文件：
`Unpaired_Baselines/fastcut_epoch10_results.md`

### 1.3 FastCUT（epoch=200）EUVP评测（已完成）
summary：
`pytorch-CycleGAN-and-pix2pix-master/results/benchmarks/euvp/external/FastCUT/epoch_200/summary.json`

（该文件内包含 PSNR/SSIM 的 95% bootstrap CI，以及 UCIQE/UIQM）

---

## 2. 当前正在运行/待完成的环节

### 2.1 FastCUT（epoch=200）UIEB评测：进行中
- 已完成 UIEB 推理输出（结果目录 `uieb_200` 已生成）
- 评测脚本正在跑（会生成下面这个 summary.json）：

目标输出：
`pytorch-CycleGAN-and-pix2pix-master/results/benchmarks/uieb/external/FastCUT/epoch_200/summary.json`

### 2.2 CUT 训练：进行中
由于 CUT 目前只训练到 epoch=2（仅有 latest），已启动继续训练：
- 训练名：`euvp_cut_full`
- 训练集：`EUVP_Unpaired`

权重目录：
`Unpaired_Baselines/checkpoints/euvp_cut_full/`

---

## 3. 你在 TraeIDE 里如何监测实验是否在跑

### 3.1 训练监测（FastCUT/CUT）
看日志是否持续追加、epoch/iters 是否增长：
- `Unpaired_Baselines/checkpoints/euvp_fastcut_full/loss_log.txt`
- `Unpaired_Baselines/checkpoints/euvp_cut_full/loss_log.txt`

看 checkpoint 是否出现/更新时间变化：
- `Unpaired_Baselines/checkpoints/euvp_fastcut_full/*_net_G.pth`
- `Unpaired_Baselines/checkpoints/euvp_cut_full/*_net_G.pth`

### 3.2 推理监测（EUVP/UIEB）
看 fake_B 目录里的图片数量是否在增长：
- EUVP：`Unpaired_Baselines/results/euvp_fastcut_full/test_<epoch>/images/fake_B`
- UIEB：`Unpaired_Baselines/results/euvp_fastcut_full/uieb_<epoch>/images/fake_B`

---

## 4. 下一步任务计划（推荐顺序）

### Step A：拿到 FastCUT(epoch200) UIEB summary
1) 等待评测完成，确认生成：
`.../uieb/external/FastCUT/epoch_200/summary.json`

2) 之后将 FastCUT 的 epoch10 与 epoch200（或选 best epoch）写入论文对比表（external baselines）。

### Step B：完成 CUT 训练并评测（补齐第二个 unpaired 强基线）
1) 等 `euvp_cut_full` 训练到 epoch200（或至少 100/150/200）
2) 对 EUVP 200 与 UIEB 进行推理：
   - EUVP：`test_<epoch>`
   - UIEB：`uieb_<epoch>`
3) 用统一脚本输出两份 summary.json：
`results/benchmarks/euvp/external/CUT/epoch_<k>/summary.json`
`results/benchmarks/uieb/external/CUT/epoch_<k>/summary.json`

### Step C：关于“SinCUT”
当前这套 CUT 官方代码仅支持 `CUT` 与 `FastCUT` 两种模式（参数 `--CUT_mode` 的 choices 为：CUT/FastCUT）。
严格意义的 **SinCUT 是单图像（single-image）训练范式**，需要不同代码与训练流程（并且通常需要“每张测试图单独训练”，不适合你的 benchmark 设定）。

建议：论文里以 **CUT + FastCUT** 作为“同级别 unpaired 强基线”，再配合你已有的 CycleGAN/MP-CycleGAN（及传统基线）即可满足“充分对比”的审稿诉求。

---

## 5. 常用命令（可直接在 TraeIDE 终端运行）

> 注意：以下命令假设你使用与现有工程一致的 Python 环境（你当前工程里已能运行这些脚本）。

### 5.1 FastCUT/CUT 推理（示例：epoch=200）
EUVP：
```bash
python <CUT_REPO>/test.py --dataroot "<EUVP_Inp>" --name euvp_fastcut_full --model cut --CUT_mode FastCUT \
  --dataset_mode single_dummy --direction AtoB --epoch 200 --phase test --num_test 200 \
  --checkpoints_dir "<.../Unpaired_Baselines/checkpoints>" --results_dir "<.../Unpaired_Baselines/results>" --gpu_ids 0
```

UIEB：
```bash
python <CUT_REPO>/test.py --dataroot "<UIEB_raw>" --name euvp_fastcut_full --model cut --CUT_mode FastCUT \
  --dataset_mode single_dummy --direction AtoB --epoch 200 --phase uieb --num_test 1000000 \
  --checkpoints_dir "<.../Unpaired_Baselines/checkpoints>" --results_dir "<.../Unpaired_Baselines/results>" --gpu_ids 0
```

### 5.2 统一评测（PSNR/SSIM + UCIQE/UIQM）
```bash
python pytorch-CycleGAN-and-pix2pix-master/scripts/evaluate_euvp_psnr_ssim.py \
  --inp_dir "<Inp>" --gtr_dir "<GTr/Ref>" --pred_dir "<PredImagesDir>" \
  --bootstrap_iters 1000 --seed 123 --save_json "<output_summary.json>"
```

