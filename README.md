# 海眸智澈

海眸智澈是一个面向水下图像增强的完整项目，围绕真实水下场景中的偏色、低对比、散射模糊和细节丢失问题，构建了“算法研究 + 指标评测 + 下游检测验证 + Web 展示演示”一体化方案。

本仓库当前上传的是适合公开共享的源码与文档版本，未包含大体积数据集、训练权重、运行产物和本地环境文件。

## 项目特点

- 支持传统方法、CycleGAN 基线和改进型 MP-CycleGAN 的统一对比
- 支持 UCIQE、UIQM、PSNR、SSIM 等多指标评测
- 支持增强前后与 YOLO 下游检测结果联动验证
- 提供本地完整演示系统，适合答辩、录屏和现场展示
- 提供静态展示版，适合部署到 Vercel 作为公开作品官网

## 主要目录

- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/`
  - 项目主体源码目录
- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/local_web_demo/`
  - Web 展示前端
- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/scripts/interactive_demo_server.py`
  - 本地完整演示服务入口
- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/scripts/underwater_competition_demo.py`
  - 推理流程、结果整理与报告生成脚本
- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/scripts/evaluate_euvp_psnr_ssim.py`
  - UCIQE、UIQM、PSNR、SSIM 评测脚本
- `关键算法与技术创新整理.md`
  - 项目关键算法、关键技术与创新点说明
- `论文附录_水下图像增强纯代码.py`
  - 水下图像增强相关纯代码整理版
- `项目总结整合文档.md`
  - 项目总结与整合文档

## 技术路线

### 传统增强

- Gray World 白平衡
- CLAHE 局部对比度增强
- Gray World + CLAHE
- Gamma 校正

### 深度学习增强

- CycleGAN 无配对图像增强
- 改进型 MP-CycleGAN

### 改进点

- 灰世界颜色约束
- 颜色统计一致性约束
- 结构梯度保持约束
- 感知损失约束
- TV 正则项

## 运行模式

### 1. 本地完整演示版

适合现场答辩、录制“真实增强 + 对比 + 检测”流程。

功能包括：

- 模型选择
- 图片增强
- 传统方法与深度模型对比
- 检测前后对照
- 指标与结果归档

核心入口：

- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/scripts/interactive_demo_server.py`

说明：

- 该模式依赖本地 Python 环境
- 若要进行真实推理，需要自行准备对应模型权重
- 本公开仓库未附带 `.pth` / `.pt` 权重文件

### 2. 静态展示版

适合部署到 Vercel，作为公开访问的作品展示页。

核心目录：

- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/local_web_demo/`

特点：

- 优先读取本地 API
- 若后端不可用，则回退到静态 JSON 数据
- 可展示首页、图表页、归档页和演示流程页

静态数据导出脚本：

- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/scripts/export_static_web_data.py`

## 快速开始

### 环境建议

- Windows
- Python 3.10 或 3.11
- 建议使用单独虚拟环境

### 安装依赖

请根据你的实际环境，在项目主目录内安装所需依赖。由于本项目包含训练、推理、评测、Web 展示等多个部分，依赖可能因使用场景不同而有所差异。

如你只需查看静态展示版，可直接进入 `local_web_demo` 目录进行部署，无需运行 Python 推理。

## 本地完整演示启动

在项目主目录下执行：

```powershell
python .\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\scripts\interactive_demo_server.py
```

启动成功后，按脚本输出访问本地地址。

如果你本机已有固定 Python 环境，也可以替换为对应解释器路径执行。

## 静态展示数据导出

在项目主目录下执行：

```powershell
python .\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\scripts\export_static_web_data.py
```

执行后会更新：

- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/local_web_demo/data/bootstrap.json`
- `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/local_web_demo/data/archive.json`

## Vercel 部署

静态展示版可部署到 Vercel。

推荐配置：

- Project Root Directory: `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/local_web_demo`
- Framework Preset: `Other`
- Build Command: 留空
- Output Directory: 留空

部署完成后即可作为作品官网展示。

## 数据集说明

本仓库未包含数据集文件。你可以从官方页面下载：

- UIEB 官方页面：[An Underwater Image Enhancement Benchmark Dataset and Beyond](https://li-chongyi.github.io/proj_benchmark)
- EUVP 官方页面：[FUnIE-GAN / EUVP Dataset](https://irvlab.cs.umn.edu/projects/funie-gan)

常见用途：

- `UIEB`：适合配对评测与参考图像指标计算
- `EUVP`：适合无配对训练、测试与增强效果对比

请遵守各数据集的原始许可、学术用途和再分发要求。

## 仓库中未包含的内容

为避免公开仓库体积过大，以下内容已被排除：

- 虚拟环境目录
- 数据集目录
- 训练权重与检测权重
- 本地运行日志
- 推理结果与归档产物
- 比赛打包文件夹
- 压缩包、安装包和大文件资源

相关忽略规则见：

- [.gitignore](file:///d:/VScode/Graduation%20project/.gitignore)

## 适用场景

- 水下图像增强课程设计
- 本科毕业设计
- 竞赛项目展示
- 算法演示与答辩录屏
- 传统方法与深度学习方法对比实验

## 参考说明

本项目在开源 CycleGAN 框架基础上进行了水下图像增强方向的适配、改进与系统化整合，并补充了本地演示、静态展示、指标评测、下游检测联动和文档材料组织。

若你需要完整复现实验，请自行准备：

- 数据集
- 模型权重
- 与本机匹配的 Python / PyTorch 环境

## 后续可补充

如果需要，我还可以继续为这个仓库补以下内容：

- 更完整的安装依赖清单
- 面向评委的精简版首页说明
- 中英双语 README
- GitHub Releases 发布说明
- 权重下载指引文档
