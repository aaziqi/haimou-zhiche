# Vercel Deployment

本目录支持部署为 Vercel 静态展示版，适用于：

- 线上公开访问作品官网
- 展示首页、图表页、归档页和演示流程页
- 保留本地完整版用于实时增强、综合对比和检测演示

## 目录说明

- `index.html`：首页，已支持自动切换为静态展示模式
- `benchmark.html`：图表页，优先读取本地 API，失败时回退到 `data/bootstrap.json`
- `archive.html`：归档页，优先读取本地 API，失败时回退到 `data/archive.json`
- `data/bootstrap.json`：静态首页与图表数据
- `data/archive.json`：静态归档数据
- `vercel.json`：Vercel 静态站点配置

## 导出最新静态数据

在项目根目录执行：

```powershell
python .\scripts\export_static_web_data.py
```

如果你使用的是指定环境，也可以改成：

```powershell
& "C:\Users\86157\anaconda3\envs\pytorch\python.exe" .\scripts\export_static_web_data.py
```

执行后会更新：

- `local_web_demo/data/bootstrap.json`
- `local_web_demo/data/archive.json`

建议在每次新增案例、更新 benchmark 或调整首页文案后重新导出一次。

## 部署步骤

1. 将 Vercel 项目根目录设置为 `local_web_demo`
2. Framework Preset 选择 `Other`
3. Build Command 留空
4. Output Directory 留空
5. 直接部署

## 双模式说明

- 本地完整版：
  通过 Python 演示服务启动，支持实时增强、综合对比、检测对照和自动归档
- Vercel 展示版：
  自动读取静态 JSON，展示已有案例、图表和系统介绍

## 注意事项

- Vercel 展示版不运行 Python 推理
- 上传增强、综合对比、实时检测按钮会自动切换为只读展示提示
- 若归档中的 `runtime` 资源路径需要线上可访问，请一并上传对应静态资源
