import csv, json
from pathlib import Path
repo = Path(r'd:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master')
summary_csv = repo / 'results' / 'benchmarks' / 'summary.csv'
archive_json = repo / 'local_web_demo' / 'runtime' / 'archive' / 'entries.json'
out_dir = repo / 'local_web_demo' / 'data'
out_dir.mkdir(parents=True, exist_ok=True)
traditional_methods = [
    {"key": "grayworld", "label": "Gray World 白平衡", "description": "传统颜色校正方法，适合快速缓解偏色问题。"},
    {"key": "clahe", "label": "CLAHE 对比度增强", "description": "传统局部对比度增强方法，能提升细节层次。"},
    {"key": "grayworld_clahe", "label": "Gray World + CLAHE", "description": "先进行白平衡，再执行局部对比度增强的组合方案。"},
    {"key": "gamma", "label": "Gamma 校正", "description": "传统非线性亮度映射方法，适合快速提亮低照度画面。"},
]
presentation_pack = {
    "project_name": "海眸智澈：面向水下机器人视觉任务的智能图像增强平台",
    "project_subtitle": "基于改进 CycleGAN 的水下图像增强与可视化评测系统",
    "track": "软件应用与开发",
    "slogan": "让浑浊水下影像变得更清晰、更自然、更可用。",
    "highlights": [
        "围绕水下偏色、低对比和细节模糊等问题，构建面向真实场景的图像增强方案。",
        "将无配对图像翻译模型升级为可展示、可评测、可批量运行的本地演示系统。",
        "同步输出 UCIQE、UIQM、PSNR、SSIM 等指标，兼顾视觉观感与工程可验证性。",
        "面向海洋监测、ROV/AUV、水下巡检与生态调查等应用场景进行展示包装。",
    ],
    "innovations": [
        "从纯算法项目升级为完整软件作品，实现模型、推理、对比、评测和展示一体化。",
        "将传统方法、原始 CycleGAN 与改进模型置于统一交互界面下进行横向验证。",
        "引入本地交互式演示界面，支持上传图片、切换模型、拖拽对比和报告导出。",
        "保留实验数据与指标榜单，形成适合软件应用展示的可信叙事链条。",
    ],
    "ppt_outline": [
        "项目背景：水下图像退化现象与应用痛点",
        "方案设计：传统方法、CycleGAN 基线与改进模型整体结构",
        "系统演示：上传图片、切换模型、前后对比、报告页与图表页",
        "实验结果：EUVP、UIEB 数据集上的客观指标与主观视觉表现",
        "应用价值：海洋监测、水下机器人、海洋牧场与应急搜救",
        "总结展望：视频增强、实时部署与更多下游视觉任务联动",
    ],
    "demo_script": [
        "第一步，展示原始水下图像，说明偏色、雾化和细节损失等典型问题。",
        "第二步，切换原始 CycleGAN 与改进模型，突出结构清晰度与色彩自然度差异。",
        "第三步，打开对比页说明传统方法虽然能提亮，但在细节保持和整体观感上存在局限。",
        "第四步，展示指标榜单与样例报告，说明作品不仅能看，还能量化验证。",
        "第五步，落脚到海洋机器人、水下巡检与生态监测等实际场景价值。",
    ],
}
rows = {}
with summary_csv.open('r', encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        rows[(row.get('dataset',''), row.get('model',''), row.get('epoch',''))] = row
def metric_value(row, key):
    raw = (row or {}).get(key, '')
    if raw in ('', None):
        return None
    try:
        return round(float(raw), 4)
    except Exception:
        return None
model_specs = [
    ("classic_cyclegan", "原 CycleGAN", "无配对图像翻译基线模型，作为原始 GAN 版本对照。", "euvp_cyclegan_full", "200", "基线模型"),
    ("improved_mpcgan", "改进模型 MPCGAN", "当前适合作为主推方案的增强模型，兼顾视觉效果与客观指标。", "euvp_mpcgan_stage2_s0", "202", "主推模型"),
    ("ablation_d", "结构约束消融增强基线", "用于展示结构损失与改进策略有效性的消融版本。", "abl_uieb_D", "20", "消融对照"),
]
models = []
for key, label, desc, model_name, epoch, badge in model_specs:
    euvp = rows.get(('euvp', model_name, epoch), {})
    uieb = rows.get(('uieb', model_name, epoch), {})
    models.append({
        'key': key,
        'label': label,
        'description': desc,
        'checkpoint_name': model_name,
        'epoch': epoch,
        'direction': 'AtoB',
        'badge': badge,
        'benchmarks': {
            'euvp': {
                'psnr': metric_value(euvp, 'psnr'),
                'ssim': metric_value(euvp, 'ssim'),
                'uciqe_pred': metric_value(euvp, 'uciqe_pred'),
                'uiqm_pred': metric_value(euvp, 'uiqm_pred'),
            },
            'uieb': {
                'psnr': metric_value(uieb, 'psnr'),
                'ssim': metric_value(uieb, 'ssim'),
                'uciqe_pred': metric_value(uieb, 'uciqe_pred'),
                'uiqm_pred': metric_value(uieb, 'uiqm_pred'),
            }
        }
    })
panels = []
for dataset, title in [('euvp', 'EUVP 数据集表现'), ('uieb', 'UIEB 泛化表现')]:
    panel_rows = []
    for model_name, epoch, label in [
        ('baseline:grayworld_clahe', '', '传统方法'),
        ('euvp_cyclegan_full', '200', '原 CycleGAN'),
        ('euvp_mpcgan_stage2_s0', '202', '改进模型'),
    ]:
        row = rows.get((dataset, model_name, epoch), {})
        panel_rows.append({
            'label': label,
            'model': model_name,
            'epoch': epoch,
            'psnr': metric_value(row, 'psnr'),
            'ssim': metric_value(row, 'ssim'),
            'uciqe_pred': metric_value(row, 'uciqe_pred'),
            'uiqm_pred': metric_value(row, 'uiqm_pred'),
        })
    panels.append({'dataset': dataset, 'title': title, 'rows': panel_rows})
archives = json.loads(archive_json.read_text(encoding='utf-8'))
bootstrap = {
    'ok': True,
    'models': models,
    'traditional_methods': traditional_methods,
    'benchmarks': panels,
    'presentation_pack': presentation_pack,
    'archives': archives[:12],
    'detectors': [],
    'runtime_mode': 'static',
}
archive_payload = {'ok': True, 'archives': archives, 'runtime_mode': 'static'}
(out_dir / 'bootstrap.json').write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding='utf-8')
(out_dir / 'archive.json').write_text(json.dumps(archive_payload, ensure_ascii=False, indent=2), encoding='utf-8')
print('ok')
