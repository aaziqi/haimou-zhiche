
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
   训练完成后，请将 `checkpoints/euvp_stage2_A_s0_long/` 目录下的 `latest_net_G_A.pth` 和 `loss_log.txt` 发回。

## 硬件建议
- 显存: >= 16GB (脚本默认 batch_size=4)
- 如果显存不足，请在 `run_train.sh` 中将 `--batch_size 4` 改为 `--batch_size 1`。
