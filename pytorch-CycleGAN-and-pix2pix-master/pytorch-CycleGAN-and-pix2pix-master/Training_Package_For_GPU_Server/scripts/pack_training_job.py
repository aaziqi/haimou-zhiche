import os
import shutil

# 配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pytorch-CycleGAN-and-pix2pix-master
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Training_Package_For_GPU_Server")
DATASET_NAME = "EUVP_Unpaired"  # 切换为EUVP数据集，数据量更大，效果更好
MODEL_NAME = "euvp_stage2_A_s0_long"  # 新的实验名


def create_package():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Creating training package at: {OUTPUT_DIR}")

    # 1. 复制核心代码
    dirs_to_copy = ["models", "options", "util", "data", "scripts"]
    files_to_copy = ["train.py", "test.py", "requirements.txt"]

    for d in dirs_to_copy:
        src = os.path.join(PROJECT_ROOT, d)
        dst = os.path.join(OUTPUT_DIR, d)
        if os.path.exists(src):
            shutil.copytree(src, dst)

    for f in files_to_copy:
        src = os.path.join(PROJECT_ROOT, f)
        dst = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # 2. 生成启动脚本 (Linux Shell)
    # 增加 --pool_size 50 以利用历史图像稳定训练
    train_sh_content = f"""#!/bin/bash
# 激活环境 (根据实际情况修改)
# source activate pytorch

# 数据集路径 (请修改为实际路径)
# 这里指向 EUVP_Unpaired，包含 trainA (水下) 和 trainB (清晰)
DATAROOT="./datasets/{DATASET_NAME}"

# 检查数据集是否存在
if [ ! -d "$DATAROOT" ]; then
    echo "Error: Dataset not found at $DATAROOT"
    echo "Please download the EUVP_Unpaired dataset and place it in the datasets folder."
    exit 1
fi

echo "Starting Full Training (400 Epochs) on EUVP Dataset..."
# 参数说明:
# --n_epochs 200: 前200轮学习率不变
# --n_epochs_decay 200: 后200轮学习率线性衰减到0
# --batch_size 4: 推荐使用大Batch Size以稳定梯度
# --pool_size 50: 使用图像缓冲区，减少震荡
# --lambda_*: 启用所有自监督损失 (Ours Config)

python train.py \\
    --dataroot "$DATAROOT" \\
    --name {MODEL_NAME} \\
    --model cycle_gan \\
    --n_epochs 200 \\
    --n_epochs_decay 200 \\
    --batch_size 4 \\
    --display_id -1 \\
    --save_epoch_freq 50 \\
    --pool_size 50 \\
    --lambda_gray 0.1 \\
    --lambda_struct 0.5 \\
    --lambda_perceptual 0.1 \\
    --lambda_color 0.0 \\
    --gpu_ids 0

echo "Training Finished!"
"""
    with open(os.path.join(OUTPUT_DIR, "run_train.sh"), "w", encoding='utf-8') as f:
        f.write(train_sh_content)

    # 3. 生成说明文档
    readme_content = f"""
# 水下图像增强模型训练任务包 (基于 EUVP 数据集)

## 任务目标
使用完整的 **EUVP (Unpaired)** 数据集，训练一个基于 MP-CycleGAN 的水下图像增强模型。
目标是跑满 400 个 epoch (约 40-50 小时，视 GPU 而定)，以获得最佳的 SSIM 指标和去雾效果。

## 目录结构
- `train.py`: 训练主程序
- `run_train.sh`: **一键启动脚本 (Linux)**
- `datasets/`: (需自行准备) 请将 EUVP_Unpaired 数据集放在此处

## 快速开始 (Linux 服务器)

1. **准备环境**:
   确保安装了 PyTorch 和 torchvision。
   ```bash
   pip install -r requirements.txt
   ```

2. **准备数据**:
   将 `EUVP_Unpaired` 数据集文件夹复制到本目录下的 `datasets/` 文件夹中。
   结构应为:
   ```
   Training_Package/
   ├── run_train.sh
   ├── datasets/
   │   └── EUVP_Unpaired/
   │       ├── trainA/ (水下图像，约 3000+ 张)
   │       └── trainB/ (清晰图像，约 3000+ 张)
   ```
   *注意：EUVP 数据集比 UIEB 更大，能提供更好的训练稳定性。*

3. **运行训练**:
   ```bash
   chmod +x run_train.sh
   ./run_train.sh
   ```

4. **结果回收**:
   训练完成后，请将 `checkpoints/{MODEL_NAME}/` 目录下的 `latest_net_G_A.pth` 和 `loss_log.txt` 发回。

## 硬件建议
- 显存: >= 16GB (脚本默认 batch_size=4)
- 如果显存不足，请在 `run_train.sh` 中将 `--batch_size 4` 改为 `--batch_size 1`。
"""
    with open(os.path.join(OUTPUT_DIR, "README_FOR_TRAINER.md"), "w", encoding='utf-8') as f:
        f.write(readme_content)

    print(f"Package updated for EUVP dataset. Please zip '{OUTPUT_DIR}' and send it to the trainer.")


if __name__ == "__main__":
    create_package()
