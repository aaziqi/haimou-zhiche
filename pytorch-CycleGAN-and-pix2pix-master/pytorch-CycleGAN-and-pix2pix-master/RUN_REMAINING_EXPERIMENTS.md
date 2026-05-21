# 跑完剩余实验的交接说明（可直接交给他人执行）

本文档面向“接手跑实验的人”。目标是：在本项目代码与已有 checkpoint 的基础上，一次性把欠缺的实验补跑完成，并产出论文所需的汇总表与图。

---

## 1. 项目位置与核心产出

项目根目录（下文默认在此目录执行命令）：

`d:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master`

核心产出文件与目录：

- 实验矩阵（每一行一条实验，包含可复制执行的命令模板）  
  `results\benchmarks\experiment_matrix.csv`
- 汇总表（论文表格/画图的数据源）  
  `results\benchmarks\summary.csv`
- 单条实验的评测结果（每条实验一份）  
  - baseline：`results\benchmarks\<dataset>\baselines\*_summary.json`、`*_per_image.csv`  
  - 模型：`results\benchmarks\<dataset>\models\<model>\epoch_<epoch>\summary.json`、`per_image.csv`
- 论文图（脚本生成/更新）  
  `docs\figures\`

---

## 2. 必备数据与 checkpoint（开跑前先核对）

### 2.1 EUVP（有监督评测用）

默认路径（脚本已写死为默认值，也可通过参数覆盖）：

- 输入：`d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp`
- GT：`d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr`

### 2.2 UIEB（跨域评测用）

本项目会自动尝试定位 UIEB 数据根目录并推断 inp/ref 子目录；也可以直接指定路径参数（推荐交接时固定写死，避免自动推断失败）。

当前机器上已使用过的路径（来自命令日志）：

- 输入：`D:\VScode\Graduation project\UIEB\raw-890-s\raw-890\raw-890`
- GT：`D:\VScode\Graduation project\UIEB\reference-890\reference-890`

### 2.3 checkpoints（模型权重）

目录：`checkpoints\`

论文 preset 会默认跑这些模型（必须存在对应 epoch 的权重，否则会跳过推理）：

- `euvp_cyclegan_full`
- `euvp_mpcgan_stage2_s0`
- `abl_uieb_A`
- `abl_uieb_B`
- `abl_uieb_C`
- `abl_uieb_D`
- `abl_uieb_E`

常见权重命名（示例）：

- `200_net_G_A.pth`、`202_net_G_A.pth`、`20_net_G_A.pth`

---

## 3. 必须用到的代码入口（只需记住这两个）

### 3.1 批量实验总控脚本（推理 + 评测 + 汇总 + 可选出图）

`scripts\euvp_auto_ss_experiment.py`

它会：

- baseline：生成输出图到 `results\benchmarks\<dataset>\baselines\<method>\images\`
- model：调用 `test.py` 生成输出图到 `results\<dataset>\<model>\test_<epoch>\images\`
- 评测：调用 `scripts\evaluate_euvp_psnr_ssim.py` 写入 `summary.json/per_image.csv`
- 汇总：更新 `results\benchmarks\summary.csv`
- 出图（可选）：更新 `docs\figures\`

### 3.2 评测脚本（一般不需要手动跑）

`scripts\evaluate_euvp_psnr_ssim.py`

---

## 4. 环境准备（Windows / PowerShell）

### 4.1 推荐：conda 一次性建环境

```powershell
cd "d:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master"
conda env create -f environment.yml
conda activate pytorch-img2img
```

### 4.2 运行中若缺包（按报错补装）

本项目的评测/出图脚本常用到这些第三方包：`opencv-python`、`pandas`、`matplotlib`、`seaborn`、`tqdm`。

如果运行时出现 `ModuleNotFoundError`，可执行：

```powershell
python -m pip install -U opencv-python pandas matplotlib seaborn tqdm
```

---

## 5. 一键把欠缺实验跑完（推荐交接执行方式）

在项目根目录执行（GPU=0，跑论文 preset：EUVP+UIEB、baseline+模型、汇总+出图）：

```powershell
cd "d:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master"

python scripts/euvp_auto_ss_experiment.py `
  --preset paper `
  --run_inference `
  --write_matrix `
  --make_figures `
  --gpu_ids 0
```

说明：

- `--preset paper` 会自动选择固定的 datasets/models/epochs（用于论文表格与图）
- `--run_inference` 若推理输出目录不存在，会自动调用 `test.py` 补跑推理
- `--write_matrix` 会生成/更新 `results\benchmarks\experiment_matrix.csv`（用于逐条补跑）
- `--make_figures` 评测完成后自动生成/更新 `docs\figures\` 下的论文图

如果只想用 CPU 跑（非常慢，不推荐），可改为：

```powershell
python scripts/euvp_auto_ss_experiment.py --preset paper --run_inference --write_matrix --make_figures --gpu_ids -1
```

---

## 6. 路径不一致时如何改（交接时最常见问题）

如果接手人机器的数据路径与本机不同，直接用参数覆盖（示例）：

```powershell
python scripts/euvp_auto_ss_experiment.py `
  --preset paper `
  --run_inference `
  --write_matrix `
  --make_figures `
  --gpu_ids 0 `
  --euvp_inp "D:\DATA\EUVP\test_samples\Inp" `
  --euvp_ref "D:\DATA\EUVP\test_samples\GTr" `
  --uieb_inp "D:\DATA\UIEB\raw" `
  --uieb_ref "D:\DATA\UIEB\reference"
```

参数说明（最常用）：

- `--euvp_inp` / `--euvp_ref`：覆盖 EUVP 测试输入与 GT
- `--uieb_inp` / `--uieb_ref`：直接指定 UIEB 输入与 GT
- `--uieb_root`：只给 UIEB 根目录，让脚本自动推断 inp/ref（不如直接指定稳定）

---

## 7. 逐条补跑（按 experiment_matrix.csv 复制命令即可）

文件：`results\benchmarks\experiment_matrix.csv`

关键字段：

- `run_cmd`：一条龙命令（推理+评测），最推荐直接复制执行
- `inference_cmd`：只推理
- `eval_cmd`：只评测（已有推理输出时用）
- `summary_json`：该条实验应产出的 summary 文件路径

执行方式：

1. 打开 `experiment_matrix.csv`
2. 找到要补跑的行（比如某个 dataset、model、epoch）
3. 复制该行的 `run_cmd` 到 PowerShell 执行

---

## 8. 如何检查“还有哪些没跑完”

最稳妥做法：以 `experiment_matrix.csv` 为准，检查每一行的 `summary_json` 是否存在。

PowerShell 检查脚本（输出缺失条目）：

```powershell
cd "d:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master"

$csv = Import-Csv "results\benchmarks\experiment_matrix.csv"
$missing = $csv | Where-Object { -not (Test-Path $_.summary_json) }
$missing | Select-Object group,dataset,kind,name,epoch,summary_json | Format-Table -AutoSize
```

---

## 9. 产出交付清单（跑完后把这些打包给原负责人）

必须交付：

- `results\benchmarks\summary.csv`
- `results\benchmarks\experiment_matrix.csv`
- `results\benchmarks\<euvp|uieb>\` 下所有 `*_summary.json`、`per_image.csv`（或整个 benchmarks 目录）
- `docs\figures\` 目录（所有 png/pdf）

---

## 10. 最后做一次一致性检查（跑完必做）

```powershell
cd "d:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master"
python -m flake8 --ignore E501 .
python -m compileall -q .
python scripts/test_before_push.py
```

如果机器没有安装 `pytest`，这是正常的；本项目当前用 `python scripts/test_before_push.py` 也能完成关键自检。

