#!/bin/bash
# 激活环境 (根据实际情况修改)
# source activate pytorch

# 数据集路径 (请修改为实际路径)
# 这里指向 EUVP_Unpaired，包含 trainA (水下) 和 trainB (清晰)
DATAROOT="./datasets/EUVP_Unpaired"

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

python train.py \
    --dataroot "$DATAROOT" \
    --name euvp_stage2_A_s0_long \
    --model cycle_gan \
    --n_epochs 200 \
    --n_epochs_decay 200 \
    --batch_size 4 \
    --display_id -1 \
    --save_epoch_freq 50 \
    --pool_size 50 \
    --lambda_gray 0.1 \
    --lambda_struct 0.5 \
    --lambda_perceptual 0.1 \
    --lambda_color 0.0 \
    --gpu_ids 0

echo "Training Finished!"
