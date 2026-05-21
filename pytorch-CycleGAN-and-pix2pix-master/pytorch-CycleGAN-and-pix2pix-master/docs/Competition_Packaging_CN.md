# 水下图像增强参赛包装方案

## 推荐赛道

建议主报 **人工智能应用**，备选 **软件应用与开发**。

- 你的核心优势是基于 CycleGAN 的水下图像增强算法，本质属于视觉生成与增强任务，赛道匹配度最高。
- 当前仓库已经具备训练、推理、评测、用户主观评价和论文素材，适合进一步包装成“算法 + 系统 + 展示”的完整作品。
- 如果后续你把演示层继续补强为完整平台界面，也可以在软件应用与开发赛道中作为强备选。

## 当前已补齐的包装层

本次新增了两部分直接面向比赛展示的能力：

- `scripts/underwater_competition_demo.py`
  - 支持单图或批量图像增强
  - 自动识别 checkpoint 中的生成器权重
  - 自动读取训练配置中的 `netG`、`norm`、`ngf`、`input_nc`、`output_nc`
  - 自动生成 `index.html`、`metrics.csv`、`summary.json`
  - 支持无参考指标 UCIQE、UIQM，以及可选参考目录下的 PSNR、SSIM
- `scripts/launch_competition_demo.ps1`
  - 提供 Windows 下一键启动方式
  - 自动寻找可用 Python 解释器
  - 更适合线下答辩、路演和快速演示

## 推荐作品定位

建议把作品统一包装为：

**“面向水下机器人与海洋监测场景的智能图像增强与可视化评测系统”**

这样可以同时覆盖三层叙事：

- 算法层：基于改进 CycleGAN 的水下图像增强
- 系统层：支持单图/批量增强、结果归档、指标统计、可视化展示
- 应用层：服务于海洋监测、ROV/AUV、水下巡检、生态调查、应急搜救

## 演示流程建议

答辩时建议用下面的顺序：

1. 先讲问题背景：水下图像普遍存在偏色、低对比、细节模糊
2. 展示原始图像与增强结果的前后对比
3. 展示 HTML 报告中的指标提升
4. 说明算法可用于批处理与工程部署
5. 落到真实场景价值：海洋牧场、水下机器人、海底文物与生态监测

## 一键运行方式

在仓库根目录执行：

```powershell
.\scripts\launch_competition_demo.ps1 `
  -InputPath "d:\VScode\Graduation project\EUVP_Unpaired\trainA\nm_0up.jpg" `
  -CheckpointName "euvp_cyclegan_full" `
  -Epoch "latest" `
  -GpuIds "-1" `
  -Overwrite
```

如果你有参考图像目录，也可以补上：

```powershell
.\scripts\launch_competition_demo.ps1 `
  -InputPath "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp" `
  -ReferenceDir "d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr" `
  -CheckpointName "euvp_cyclegan_full" `
  -Epoch "latest" `
  -GpuIds "-1" `
  -Overwrite
```

生成结果默认保存在：

`competition_demo_outputs/national_award_showcase/`

其中最重要的是：

- `index.html`：答辩展示页
- `metrics.csv`：指标汇总表
- `summary.json`：结构化摘要，可继续接入更多展示页面或前端系统

## 冲击国奖建议

接下来最值得继续补强的方向有四个：

- **场景化**：把应用场景从“图像增强”扩展到“水下机器人视觉可用性提升”
- **对比化**：补充传统方法、原始 CycleGAN、改进方法三组对比
- **系统化**：把当前 HTML 展示页继续升级为本地 Web 演示平台
- **成果化**：准备答辩 PPT、演示视频、软件著作权材料与论文摘要

## 建议你后续继续完善的功能

- 增加视频增强演示
- 增加实时摄像头/视频流模拟接口
- 增加更多公开数据集对比
- 增加用户主观评价结果汇总页
- 增加模型版本管理与实验看板

这样你最终交付的不再只是“一个模型”，而是“一个可展示、可答辩、可落地的智能视觉系统”。
