from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import mimetypes
import os
import shutil
import sys
import time
import threading
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "local_web_demo"
RUNTIME_ROOT = WEB_ROOT / "runtime"
ARCHIVE_ROOT = RUNTIME_ROOT / "archive"
ARCHIVE_INDEX = ARCHIVE_ROOT / "entries.json"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
BENCHMARK_SUMMARY = REPO_ROOT / "results" / "benchmarks" / "summary.csv"
DEFAULT_PROJECT_ROOT = REPO_ROOT.parents[1] if len(REPO_ROOT.parents) > 1 else REPO_ROOT


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


demo = load_module("underwater_competition_demo_web", SCRIPTS_ROOT / "underwater_competition_demo.py")
auto_exp = load_module("euvp_auto_ss_experiment_web", SCRIPTS_ROOT / "euvp_auto_ss_experiment.py")
METRICS = demo.load_metric_functions()


TRADITIONAL_METHODS = [
    {
        "key": "grayworld",
        "label": "Gray World 白平衡",
        "description": "传统颜色校正方法，适合快速缓解偏色问题。",
    },
    {
        "key": "clahe",
        "label": "CLAHE 对比度增强",
        "description": "传统局部对比度增强方法，能提升细节层次。",
    },
    {
        "key": "grayworld_clahe",
        "label": "Gray World + CLAHE",
        "description": "先进行白平衡，再执行局部对比度增强的组合方案。",
    },
    {
        "key": "gamma",
        "label": "Gamma 校正",
        "description": "传统非线性亮度映射方法，适合快速提亮低照度画面。",
    },
]
PRESENTATION_PACK = {
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
def resolve_project_root() -> Path:
    for candidate in [DEFAULT_PROJECT_ROOT, REPO_ROOT.parent, REPO_ROOT]:
        if (candidate / "datasets").exists() or (candidate / "results").exists():
            return candidate
    return DEFAULT_PROJECT_ROOT


def load_names_from_data_yaml(data_yaml: Path) -> list[str]:
    if not data_yaml.exists():
        return []
    try:
        import yaml
    except Exception:
        return []
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = d.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        out: list[str] = []
        for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            out.append(str(names[k]))
        return out
    return []


def build_detector_presets() -> list[dict]:
    project_root = resolve_project_root()
    weights_env = os.environ.get("DEMO_DETECT_WEIGHTS", "").strip()
    data_env = os.environ.get("DEMO_DETECT_DATA_YAML", "").strip()

    weights_candidates = [
        Path(weights_env) if weights_env else None,
        project_root / "results" / "yolo_urpc_train" / "yolov8n_urpc_final" / "weights" / "best.pt",
        project_root / "yolov8n.pt",
        project_root / "yolo11n.pt",
        REPO_ROOT / "yolo11n.pt",
    ]
    data_candidates = [
        Path(data_env) if data_env else None,
        project_root / "datasets" / "URPC_optical" / "yolo_new" / "data.yaml",
        project_root / "datasets" / "URPC_optical" / "yolo" / "data.yaml",
    ]

    weights_path = next((p.resolve() for p in weights_candidates if isinstance(p, Path) and p.exists()), None)
    data_yaml = next((p.resolve() for p in data_candidates if isinstance(p, Path) and p.exists()), None)
    names = load_names_from_data_yaml(data_yaml) if data_yaml else []

    presets: list[dict] = []
    if weights_path is not None:
        presets.append(
            {
                "key": "default_detector",
                "label": "下游检测器 (YOLO)",
                "description": "在增强结果上运行下游检测器，输出可视化图与检测结果 JSON。",
                "weights": str(weights_path),
                "data_yaml": str(data_yaml) if data_yaml else "",
                "names": names,
                "default_conf": 0.25,
                "default_imgsz": 640,
            }
        )
    return presets


DETECTOR_PRESETS = build_detector_presets()
_YOLO_MODEL_LOCK = threading.Lock()
_YOLO_MODEL_CACHE: dict[str, object] = {}


def pick_detector(detector_key: str) -> dict:
    if not DETECTOR_PRESETS:
        raise RuntimeError("检测器未就绪，请准备权重 `best.pt`，并设置环境变量 `DEMO_DETECT_WEIGHTS` / `DEMO_DETECT_DATA_YAML`。")
    for item in DETECTOR_PRESETS:
        if item["key"] == detector_key:
            return item
    return DETECTOR_PRESETS[0]


def get_yolo_model(weights_path: Path):
    key = str(weights_path.resolve())
    with _YOLO_MODEL_LOCK:
        cached = _YOLO_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        from ultralytics import YOLO
        model = YOLO(key)
        _YOLO_MODEL_CACHE[key] = model
        return model


def web_url_to_path(url: str) -> Path:
    parsed = urlparse(url)
    rel = unquote(parsed.path).lstrip("/")
    p = (WEB_ROOT / rel).resolve()
    web_root = WEB_ROOT.resolve()
    if web_root not in p.parents and p != web_root:
        raise ValueError("非法资源路径。")
    return p


def run_detector_on_image(
    *,
    input_path: Path,
    output_vis_path: Path,
    output_json_path: Path,
    detector: dict,
    conf: float,
    imgsz: int,
):
    from PIL import Image, ImageDraw, ImageFont

    weights_path = Path(detector["weights"]).resolve()
    model = get_yolo_model(weights_path)

    names = detector.get("names") or []
    if not names and getattr(model, "names", None):
        model_names = getattr(model, "names")
        if isinstance(model_names, dict):
            names = [str(model_names[k]) for k in sorted(model_names.keys())]
        elif isinstance(model_names, list):
            names = [str(x) for x in model_names]

    image = Image.open(input_path).convert("RGB")
    results = model.predict(image, conf=float(conf), imgsz=int(imgsz), verbose=False)
    dets: list[dict] = []
    boxes = []
    if results:
        res = results[0]
        if getattr(res, "boxes", None) is not None:
            xyxy = res.boxes.xyxy
            cls = res.boxes.cls
            confs = res.boxes.conf
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [float(x) for x in xyxy[i].tolist()]
                cls_id = int(cls[i].item())
                score = float(confs[i].item())
                boxes.append((x1, y1, x2, y2, cls_id, score))
                dets.append(
                    {
                        "cls_id": cls_id,
                        "cls_name": names[cls_id] if 0 <= cls_id < len(names) else str(cls_id),
                        "conf": round(score, 6),
                        "x1": round(x1, 3),
                        "y1": round(y1, 3),
                        "x2": round(x2, 3),
                        "y2": round(y2, 3),
                    }
                )

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for x1, y1, x2, y2, cls_id, score in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 90), width=3)
        label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
        label = f"{label} {score:.2f}"
        if font is not None:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            tx, ty = x1, max(0, y1 - th - 3)
            draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(0, 0, 0))
            draw.text((tx + 3, ty + 2), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((x1 + 3, y1 + 3), label, fill=(255, 255, 255))

    output_vis_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(output_vis_path)
    output_json_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "width": image.width,
                "height": image.height,
                "detector": {
                    "key": detector["key"],
                    "label": detector.get("label", ""),
                    "weights": detector.get("weights", ""),
                    "data_yaml": detector.get("data_yaml", ""),
                },
                "conf": float(conf),
                "imgsz": int(imgsz),
                "detections": dets,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dets


def build_detection_vs_image(
    *,
    raw_vis_path: Path,
    enhanced_vis_path: Path,
    out_path: Path,
    titles: tuple[str, str] = ("RAW (Pred)", "Enhanced (Pred)"),
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    left = Image.open(raw_vis_path).convert("RGB")
    right = Image.open(enhanced_vis_path).convert("RGB")
    if left.size != right.size:
        right = right.resize(left.size, resample=Image.BICUBIC)
    w, h = left.size
    top_h = 32
    out = Image.new("RGB", (w * 2, h + top_h), (255, 255, 255))
    out.paste(left, (0, top_h))
    out.paste(right, (w, top_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, t in enumerate(titles):
        x = i * w + 8
        y = 8
        if font is not None:
            draw.text((x, y), t, fill=(0, 0, 0), font=font)
        else:
            draw.text((x, y), t, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--gpu_ids", default="-1")
    return parser.parse_args()


def ensure_runtime_root() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_INDEX.exists():
        ARCHIVE_INDEX.write_text("[]", encoding="utf-8")


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: SimpleHTTPRequestHandler, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
    json_response(handler, {"ok": False, "error": message}, status)


def read_json_request(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    data = handler.rfile.read(length) if length > 0 else b"{}"
    return json.loads(data.decode("utf-8"))


def safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "image"


def decode_image_payload(filename: str, payload: str, target_dir: Path) -> Path:
    if "," in payload:
        payload = payload.split(",", 1)[1]
    raw = base64.b64decode(payload)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        suffix = ".png"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{safe_slug(Path(filename).stem)}{suffix}"
    target_path.write_bytes(raw)
    return target_path


def to_web_url(path: Path) -> str:
    relative = path.resolve().relative_to(WEB_ROOT.resolve())
    return "/" + quote(relative.as_posix())


def load_archive_entries() -> list[dict]:
    if not ARCHIVE_INDEX.exists():
        return []
    try:
        return json.loads(ARCHIVE_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_archive_entries(entries: list[dict]) -> None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_INDEX.write_text(json.dumps(entries[:60], ensure_ascii=False, indent=2), encoding="utf-8")


def add_archive_entry(entry: dict) -> None:
    entries = load_archive_entries()
    entries.insert(0, entry)
    save_archive_entries(entries)


def read_benchmark_rows() -> list[dict]:
    rows: list[dict] = []
    if not BENCHMARK_SUMMARY.exists():
        return rows
    with BENCHMARK_SUMMARY.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def benchmark_lookup() -> dict[tuple[str, str, str], dict]:
    table = {}
    for row in read_benchmark_rows():
        table[(row.get("dataset", ""), row.get("model", ""), row.get("epoch", ""))] = row
    return table


def metric_value(row: dict, key: str) -> float | None:
    raw = (row or {}).get(key, "")
    if raw in {"", None}:
        return None
    try:
        return round(float(raw), 4)
    except ValueError:
        return None


def build_model_presets() -> list[dict]:
    summary = benchmark_lookup()
    presets = [
        {
            "key": "classic_cyclegan",
            "label": "原 CycleGAN",
            "description": "无配对图像翻译基线模型，作为原始 GAN 版本对照。",
            "checkpoint_name": "euvp_cyclegan_full",
            "epoch": "200",
            "direction": "AtoB",
            "badge": "基线模型",
        },
        {
            "key": "improved_mpcgan",
            "label": "改进模型 MPCGAN",
            "description": "当前适合作为主推方案的增强模型，兼顾视觉效果与客观指标。",
            "checkpoint_name": "euvp_mpcgan_stage2_s0",
            "epoch": "202",
            "direction": "AtoB",
            "badge": "主推模型",
        },
        {
            "key": "ablation_d",
            "label": "结构约束消融增强基线",
            "description": "用于展示结构损失与改进策略有效性的消融版本。",
            "checkpoint_name": "abl_uieb_D",
            "epoch": "20",
            "direction": "AtoB",
            "badge": "消融对照",
        },
    ]
    available = []
    for item in presets:
        checkpoint_dir = CHECKPOINTS_DIR / item["checkpoint_name"]
        if not checkpoint_dir.exists():
            continue
        suffix = demo.detect_model_suffix(checkpoint_dir, item["epoch"], item["direction"])
        if suffix is None:
            continue
        euvp = summary.get(("euvp", item["checkpoint_name"], item["epoch"]), {})
        uieb = summary.get(("uieb", item["checkpoint_name"], item["epoch"]), {})
        item["benchmarks"] = {
            "euvp": {
                "psnr": metric_value(euvp, "psnr"),
                "ssim": metric_value(euvp, "ssim"),
                "uciqe_pred": metric_value(euvp, "uciqe_pred"),
                "uiqm_pred": metric_value(euvp, "uiqm_pred"),
            },
            "uieb": {
                "psnr": metric_value(uieb, "psnr"),
                "ssim": metric_value(uieb, "ssim"),
                "uciqe_pred": metric_value(uieb, "uciqe_pred"),
                "uiqm_pred": metric_value(uieb, "uiqm_pred"),
            },
        }
        available.append(item)
    return available


def build_benchmark_panels() -> list[dict]:
    rows = benchmark_lookup()
    euvp_entries = [
        ("baseline:grayworld_clahe", "", "传统方法"),
        ("euvp_cyclegan_full", "200", "原 CycleGAN"),
        ("euvp_mpcgan_stage2_s0", "202", "改进模型"),
    ]
    uieb_entries = [
        ("baseline:grayworld_clahe", "", "传统方法"),
        ("euvp_cyclegan_full", "200", "原 CycleGAN"),
        ("euvp_mpcgan_stage2_s0", "202", "改进模型"),
    ]
    panels = []
    for dataset, entries, title in [
        ("euvp", euvp_entries, "EUVP 数据集表现"),
        ("uieb", uieb_entries, "UIEB 泛化表现"),
    ]:
        panel_rows = []
        for model, epoch, label in entries:
            row = rows.get((dataset, model, epoch), {})
            panel_rows.append(
                {
                    "label": label,
                    "model": model,
                    "epoch": epoch,
                    "psnr": metric_value(row, "psnr"),
                    "ssim": metric_value(row, "ssim"),
                    "uciqe_pred": metric_value(row, "uciqe_pred"),
                    "uiqm_pred": metric_value(row, "uiqm_pred"),
                }
            )
        panels.append({"dataset": dataset, "title": title, "rows": panel_rows})
    return panels


def build_archive_entry_for_enhance(job_id: str, result: dict) -> dict:
    return {
        "id": job_id,
        "type": "enhance",
        "title": f"{result['model']['label']} 单图增强案例",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_label": result["model"]["label"],
        "summary": f"UCIQE 提升 {result.get('delta_uciqe', '?')}，UIQM 提升 {result.get('delta_uiqm', '?')}",
        "input_url": result["input_url"],
        "cover_url": result["output_url"],
        "report_url": result["report_url"],
        "metrics_url": result["metrics_url"],
        "tags": ["单图增强", result["model"]["badge"]],
    }


def build_archive_entry_for_compare(job_id: str, result: dict) -> dict:
    improved_variant = next((item for item in result["variants"] if item["badge"] != "传统方法" and "改进" in item["label"]), result["variants"][-1])
    return {
        "id": job_id,
        "type": "compare",
        "title": "传统方法与改进模型综合对比",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_label": improved_variant["label"],
        "summary": f"{improved_variant['label']} 与传统方法、原始 CycleGAN 的同图横向对比。",
        "input_url": result["input_url"],
        "cover_url": improved_variant["image_url"],
        "report_url": result["compare_report_url"],
        "metrics_url": improved_variant.get("report_url", ""),
        "tags": ["综合对比", improved_variant["badge"]],
    }


def build_archive_entry_for_detect(job_id: str, *, image_url: str, vis_url: str, json_url: str, detector: dict, num_dets: int, report_url: str) -> dict:
    return {
        "id": job_id,
        "type": "detect",
        "title": "下游检测",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_label": detector.get("label", "下游检测器"),
        "summary": f"检测目标数 {num_dets}，结果来自增强图像上的下游检测器。",
        "input_url": image_url,
        "cover_url": vis_url,
        "report_url": report_url,
        "metrics_url": json_url,
        "tags": ["下游检测"],
    }


def build_archive_entry_for_detect_compare(
    job_id: str,
    *,
    raw_image_url: str,
    enhanced_image_url: str,
    compare_vis_url: str,
    raw_json_url: str,
    enhanced_json_url: str,
    detector: dict,
    raw_num_dets: int,
    enhanced_num_dets: int,
    report_url: str,
) -> dict:
    delta = enhanced_num_dets - raw_num_dets
    delta_text = f"{delta:+d}" if isinstance(delta, int) else str(delta)
    return {
        "id": job_id,
        "type": "detect_compare",
        "title": "下游检测对照",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_label": detector.get("label", "下游检测器"),
        "summary": f"原图 {raw_num_dets} vs 增强图 {enhanced_num_dets}，差值 {delta_text}。",
        "input_url": raw_image_url,
        "cover_url": compare_vis_url,
        "report_url": report_url,
        "metrics_url": enhanced_json_url,
        "tags": ["下游检测", "对照"],
        "extra": {
            "raw_image_url": raw_image_url,
            "enhanced_image_url": enhanced_image_url,
            "raw_json_url": raw_json_url,
            "enhanced_json_url": enhanced_json_url,
        },
    }


def build_detect_compare_report(
    job_root: Path,
    *,
    raw_image_url: str,
    enhanced_image_url: str,
    raw_vis_url: str,
    enhanced_vis_url: str,
    compare_vis_url: str,
    raw_json_url: str,
    enhanced_json_url: str,
    detector: dict,
    raw_dets: list[dict],
    enhanced_dets: list[dict],
) -> str:
    raw_rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{d.get('cls_name','')}</td><td>{d.get('conf','')}</td><td>{d.get('x1','')}</td><td>{d.get('y1','')}</td><td>{d.get('x2','')}</td><td>{d.get('y2','')}</td></tr>"
        for i, d in enumerate(raw_dets[:200])
    )
    enh_rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{d.get('cls_name','')}</td><td>{d.get('conf','')}</td><td>{d.get('x1','')}</td><td>{d.get('y1','')}</td><td>{d.get('x2','')}</td><td>{d.get('y2','')}</td></tr>"
        for i, d in enumerate(enhanced_dets[:200])
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>下游测照报?/title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    img {{ width: 100%; max-height: 520px; object-fit: contain; border: 1px solid #ddd; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 8px 10px; font-size: 14px; text-align: left; }}
    th {{ background: #fafafa; }}
    .meta {{ margin-top: 12px; color: #555; line-height: 1.8; }}
    .links a {{ margin-right: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }}
    .card {{ border: 1px solid #eee; border-radius: 12px; padding: 14px; }}
    h3 {{ margin: 0 0 10px; }}
  </style>
</head>
<body>
  <h2>下游测照报?/h2>
  <div class="meta">测器：{detector.get('label','下游测器')} · 原图盠数：{len(raw_dets)} · 增强盠数：{len(enhanced_dets)}</div>
  <div class="links meta">
    <a href="{raw_json_url}" target="_blank" rel="noopener">下载 raw_detections.json</a>
    <a href="{enhanced_json_url}" target="_blank" rel="noopener">下载 enhanced_detections.json</a>
  </div>

  <div class="meta">对照图（左：原图测，右：增强测）</div>
  <img src="{compare_vis_url}" alt="测照图">

  <div class="grid">
    <div class="card">
      <h3>原图?/h3>
      <div class="row">
        <div><div class="meta">原图</div><img src="{raw_image_url}" alt="原图"></div>
        <div><div class="meta">原图测可视化</div><img src="{raw_vis_url}" alt="原图测可视化"></div>
      </div>
      <table>
        <thead><tr><th>#</th><th>类别</th><th>罿?/th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr></thead>
        <tbody>{raw_rows if raw_rows else "<tr><td colspan='7'>测到盠</td></tr>"}</tbody>
      </table>
    </div>
    <div class="card">
      <h3>增强?/h3>
      <div class="row">
        <div><div class="meta">增强?/div><img src="{enhanced_image_url}" alt="增强?></div>
        <div><div class="meta">增强测可视化</div><img src="{enhanced_vis_url}" alt="增强测可视化"></div>
      </div>
      <table>
        <thead><tr><th>#</th><th>类别</th><th>罿?/th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr></thead>
        <tbody>{enh_rows if enh_rows else "<tr><td colspan='7'>测到盠</td></tr>"}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    report_path = job_root / "index.html"
    report_path.write_text(html, encoding="utf-8")
    return to_web_url(report_path)


def build_detect_report(job_root: Path, *, image_url: str, vis_url: str, json_url: str, detector: dict, dets: list[dict]) -> str:
    rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{d.get('cls_name','')}</td><td>{d.get('conf','')}</td><td>{d.get('x1','')}</td><td>{d.get('y1','')}</td><td>{d.get('x2','')}</td><td>{d.get('y2','')}</td></tr>"
        for i, d in enumerate(dets[:200])
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>下游测报?/title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    img {{ width: 100%; max-height: 520px; object-fit: contain; border: 1px solid #ddd; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 8px 10px; font-size: 14px; text-align: left; }}
    th {{ background: #fafafa; }}
    .meta {{ margin-top: 12px; color: #555; line-height: 1.8; }}
    .links a {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <h2>下游测报?/h2>
  <div class="meta">测器：{detector.get('label','下游测器')} · 盠数：{len(dets)}</div>
  <div class="links meta">
    <a href="{json_url}" target="_blank" rel="noopener">下载 detections.json</a>
  </div>
  <div class="row">
    <div>
      <div class="meta">输入图（增强结果?/div>
      <img src="{image_url}" alt="输入?>
    </div>
    <div>
      <div class="meta">测可视化</div>
      <img src="{vis_url}" alt="测可视化">
    </div>
  </div>
  <table>
    <thead>
      <tr><th>#</th><th>类别</th><th>罿?/th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr>
    </thead>
    <tbody>
      {rows if rows else "<tr><td colspan='7'>测到盠</td></tr>"}
    </tbody>
  </table>
</body>
</html>
"""
    report_path = job_root / "index.html"
    report_path.write_text(html, encoding="utf-8")
    return to_web_url(report_path)


def pick_model(model_key: str) -> dict:
    for preset in build_model_presets():
        if preset["key"] == model_key:
            return preset
    raise KeyError(f"有到模型? {model_key}")


def compute_metrics(input_path: Path, output_path: Path) -> dict:
    if METRICS is None:
        return {}
    input_bgr = METRICS.read_image(input_path)
    output_bgr = METRICS.read_image(output_path)
    input_uciqe = METRICS.compute_uciqe(input_bgr)
    output_uciqe = METRICS.compute_uciqe(output_bgr)
    input_uiqm = METRICS.compute_uiqm(input_bgr)
    output_uiqm = METRICS.compute_uiqm(output_bgr)
    return {
        "input_uciqe": demo.safe_round(input_uciqe),
        "output_uciqe": demo.safe_round(output_uciqe),
        "delta_uciqe": demo.safe_round(output_uciqe - input_uciqe),
        "input_uiqm": demo.safe_round(input_uiqm),
        "output_uiqm": demo.safe_round(output_uiqm),
        "delta_uiqm": demo.safe_round(output_uiqm - input_uiqm),
    }


def run_single_model(job_root: Path, input_path: Path, preset: dict, gpu_ids: str) -> dict:
    report_root = job_root / preset["key"]
    input_dir = input_path.parent
    command, raw_result_dir = demo.build_test_command(
        input_dir=input_dir,
        output_dir=report_root,
        checkpoint_name=preset["checkpoint_name"],
        epoch=preset["epoch"],
        direction=preset["direction"],
        num_test=1,
    )
    started = time.perf_counter()
    demo.run_inference(command, gpu_ids)
    elapsed = time.perf_counter() - started
    prediction_dir = raw_result_dir / "images"
    assets_dir = report_root / "assets"
    records = demo.build_records(input_dir, prediction_dir, None, assets_dir)
    if not records:
        raise RuntimeError("模型推理已完成，但未生成对应的结果记录。")
    demo.write_metrics_csv(records, report_root / "metrics.csv")
    demo.build_html_report(
        records=records,
        output_dir=report_root,
        title=f"{preset['label']} 单图增强报告",
        track_name="软件应用与开发",
        checkpoint_name=preset["checkpoint_name"],
        epoch=preset["epoch"],
        raw_result_dir=raw_result_dir,
    )
    record = records[0]
    payload = asdict(record)
    payload["input_url"] = to_web_url(report_root / record.input_file)
    payload["output_url"] = to_web_url(report_root / record.output_file)
    payload["report_url"] = to_web_url(report_root / "index.html")
    payload["metrics_url"] = to_web_url(report_root / "metrics.csv")
    payload["processing_seconds"] = round(elapsed, 3)
    payload["model"] = preset
    return payload


def apply_traditional_method(method_key: str, input_path: Path, output_path: Path) -> None:
    image = auto_exp._read_bgr(input_path)
    if method_key == "grayworld":
        result = auto_exp._gray_world_wb(image)
    elif method_key == "clahe":
        result = auto_exp._clahe_lab(image)
    elif method_key == "grayworld_clahe":
        result = auto_exp._clahe_lab(auto_exp._gray_world_wb(image))
    elif method_key == "gamma":
        result = auto_exp._gamma(image)
    else:
        raise ValueError(f"期传统方法: {method_key}")
    auto_exp._write_png(output_path, result)


def traditional_method_meta(method_key: str) -> dict:
    for item in TRADITIONAL_METHODS:
        if item["key"] == method_key:
            return item
    raise KeyError(f"有到传统方? {method_key}")


def build_compare_report(job_root: Path, input_url: str, variants: list[dict], benchmark_panels: list[dict]) -> str:
    cards = []
    for variant in variants:
        metrics = variant.get("metrics", {})
        cards.append(
            f"""
            <section class="compare-card">
                <div class="compare-head">
                    <div>
                        <div class="compare-badge">{variant['badge']}</div>
                        <h2>{variant['label']}</h2>
                    </div>
                    <div class="compare-desc">{variant['description']}</div>
                </div>
                <div class="compare-grid">
                    <div class="compare-panel">
                        <div class="compare-title">输入图像</div>
                        <img src="{input_url}" alt="input">
                    </div>
                    <div class="compare-panel">
                        <div class="compare-title">输出结果</div>
                        <img src="{variant['image_url']}" alt="{variant['label']}">
                    </div>
                </div>
                <div class="metric-row">
                    <div class="metric-box"><span>输出 UCIQE</span><strong>{metrics.get('output_uciqe', '?')}</strong></div>
                    <div class="metric-box"><span>输出 UIQM</span><strong>{metrics.get('output_uiqm', '?')}</strong></div>
                    <div class="metric-box"><span>UCIQE 提升</span><strong>{metrics.get('delta_uciqe', '?')}</strong></div>
                    <div class="metric-box"><span>UIQM 提升</span><strong>{metrics.get('delta_uiqm', '?')}</strong></div>
                </div>
            </section>
            """
        )
    bench_blocks = []
    for panel in benchmark_panels:
        rows_html = "".join(
            f"<tr><td>{row['label']}</td><td>{row['psnr'] or '?'}</td><td>{row['ssim'] or '?'}</td><td>{row['uciqe_pred'] or '?'}</td><td>{row['uiqm_pred'] or '?'}</td></tr>"
            for row in panel["rows"]
        )
        bench_blocks.append(
            f"""
            <section class="bench-card">
                <h3>{panel['title']}</h3>
                <table>
                    <thead><tr><th>方法</th><th>PSNR</th><th>SSIM</th><th>UCIQE</th><th>UIQM</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </section>
            """
        )
    html_text = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>水下图像增强综合对比</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #07111f; color: #eef5ff; }}
            .wrap {{ width: min(1240px, calc(100% - 40px)); margin: 0 auto; padding: 28px 0 48px; }}
            .hero {{ padding: 28px; border-radius: 24px; background: linear-gradient(135deg, #18457d, #0a1830); border: 1px solid rgba(120,180,255,0.2); }}
            .hero h1 {{ margin: 0 0 10px; font-size: 34px; }}
            .hero p {{ margin: 0; line-height: 1.8; color: #adc4e4; }}
            .compare-card, .bench-card {{ margin-top: 22px; padding: 22px; border-radius: 22px; background: rgba(11, 24, 44, 0.88); border: 1px solid rgba(120,180,255,0.14); }}
            .compare-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: center; margin-bottom: 18px; }}
            .compare-badge {{ display: inline-block; margin-bottom: 10px; padding: 6px 12px; border-radius: 999px; background: rgba(75,195,255,0.18); color: #d8f5ff; }}
            .compare-head h2 {{ margin: 0; font-size: 24px; }}
            .compare-desc {{ max-width: 480px; color: #a8bfdc; line-height: 1.7; }}
            .compare-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
            .compare-panel {{ padding: 14px; border-radius: 18px; background: rgba(255,255,255,0.03); }}
            .compare-title {{ margin-bottom: 10px; color: #9cb4d3; }}
            .compare-panel img {{ width: 100%; height: 280px; object-fit: contain; border-radius: 14px; background: rgba(255,255,255,0.02); }}
            .metric-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }}
            .metric-box {{ padding: 14px; border-radius: 16px; background: rgba(255,255,255,0.03); }}
            .metric-box span {{ display: block; font-size: 13px; color: #98b1cf; margin-bottom: 8px; }}
            .metric-box strong {{ font-size: 22px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left; }}
            th {{ color: #9cb4d3; font-weight: 600; }}
            @media (max-width: 840px) {{
                .compare-head {{ flex-direction: column; align-items: flex-start; }}
            }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <section class="hero">
                <h1>传统方法 vs 原 CycleGAN vs 改进模型</h1>
                <p>该页面面向软件应用展示场景，统一展示传统增强方法、原始无配对增强基线与改进模型的单图对比结果，并结合 EUVP / UIEB 的基准数据说明方法演进价值。</p>
            </section>
            {''.join(cards)}
            {''.join(bench_blocks)}
        </div>
    </body>
    </html>
    """
    report_path = job_root / "compare_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return to_web_url(report_path)


def build_compare_job(input_path: Path, traditional_key: str, improved_key: str, gpu_ids: str) -> dict:
    job_id = f"compare_{uuid.uuid4().hex[:10]}"
    job_root = RUNTIME_ROOT / "compare_jobs" / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    copied_input = job_root / "input" / input_path.name
    copied_input.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, copied_input)
    input_url = to_web_url(copied_input)

    traditional_meta = traditional_method_meta(traditional_key)
    traditional_output = job_root / "traditional" / f"{copied_input.stem}_{traditional_key}.png"
    apply_traditional_method(traditional_key, copied_input, traditional_output)
    traditional_payload = {
        "key": traditional_key,
        "label": traditional_meta["label"],
        "badge": "传统方法",
        "description": traditional_meta["description"],
        "image_url": to_web_url(traditional_output),
        "metrics": compute_metrics(copied_input, traditional_output),
    }

    classic_model = pick_model("classic_cyclegan")
    classic_payload = run_single_model(job_root / "classic", copied_input, classic_model, gpu_ids)
    improved_model = pick_model(improved_key)
    improved_payload = run_single_model(job_root / "improved", copied_input, improved_model, gpu_ids)

    variants = [
        traditional_payload,
        {
            "key": classic_model["key"],
            "label": classic_model["label"],
            "badge": classic_model["badge"],
            "description": classic_model["description"],
            "image_url": classic_payload["output_url"],
            "metrics": {
                "output_uciqe": classic_payload["output_uciqe"],
                "output_uiqm": classic_payload["output_uiqm"],
                "delta_uciqe": classic_payload["delta_uciqe"],
                "delta_uiqm": classic_payload["delta_uiqm"],
            },
            "report_url": classic_payload["report_url"],
        },
        {
            "key": improved_model["key"],
            "label": improved_model["label"],
            "badge": improved_model["badge"],
            "description": improved_model["description"],
            "image_url": improved_payload["output_url"],
            "metrics": {
                "output_uciqe": improved_payload["output_uciqe"],
                "output_uiqm": improved_payload["output_uiqm"],
                "delta_uciqe": improved_payload["delta_uciqe"],
                "delta_uiqm": improved_payload["delta_uiqm"],
            },
            "report_url": improved_payload["report_url"],
        },
    ]

    compare_report_url = build_compare_report(job_root, input_url, variants, build_benchmark_panels())
    return {
        "job_id": job_id,
        "input_url": input_url,
        "variants": variants,
        "compare_report_url": compare_report_url,
        "benchmark_panels": build_benchmark_panels(),
    }


def bootstrap_payload() -> dict:
    return {
        "ok": True,
        "models": build_model_presets(),
        "traditional_methods": TRADITIONAL_METHODS,
        "benchmarks": build_benchmark_panels(),
        "presentation_pack": PRESENTATION_PACK,
        "archives": load_archive_entries()[:12],
        "detectors": DETECTOR_PRESETS,
    }


class DemoRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, gpu_ids: str = "-1", **kwargs):
        self.gpu_ids = gpu_ids
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/bootstrap":
            json_response(self, bootstrap_payload())
            return
        if parsed.path == "/api/archive":
            json_response(self, {"ok": True, "archives": load_archive_entries()})
            return
        if parsed.path == "/api/health":
            json_response(self, {"ok": True})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/enhance":
                payload = read_json_request(self)
                filename = payload.get("filename", "upload.png")
                image_data = payload.get("image", "")
                model_key = payload.get("model_key", "improved_mpcgan")
                if not image_data:
                    error_response(self, "未收到上传图片。")
                    return
                job_id = f"enhance_{uuid.uuid4().hex[:10]}"
                upload_root = RUNTIME_ROOT / "enhance_jobs" / job_id / "upload"
                input_path = decode_image_payload(filename, image_data, upload_root)
                preset = pick_model(model_key)
                result = run_single_model(RUNTIME_ROOT / "enhance_jobs" / job_id / "result", input_path, preset, self.gpu_ids)
                add_archive_entry(build_archive_entry_for_enhance(job_id, result))
                json_response(self, {"ok": True, "job_id": job_id, "result": result})
                return
            if parsed.path == "/api/detect":
                payload = read_json_request(self)
                image_url = payload.get("image_url", "")
                detector_key = payload.get("detector_key", "default_detector")
                conf = float(payload.get("conf", 0.25))
                imgsz = int(payload.get("imgsz", 640))
                if not image_url:
                    error_response(self, "缺少 `image_url`。")
                    return
                input_path = web_url_to_path(image_url)
                if not input_path.exists():
                    error_response(self, "`image_url` 对应文件不存在。")
                    return
                detector = pick_detector(detector_key)
                job_id = f"detect_{uuid.uuid4().hex[:10]}"
                job_root = RUNTIME_ROOT / "detect_jobs" / job_id
                job_root.mkdir(parents=True, exist_ok=True)
                vis_path = job_root / "det_vis.png"
                json_path = job_root / "detections.json"
                dets = run_detector_on_image(
                    input_path=input_path,
                    output_vis_path=vis_path,
                    output_json_path=json_path,
                    detector=detector,
                    conf=conf,
                    imgsz=imgsz,
                )
                report_url = build_detect_report(
                    job_root,
                    image_url=image_url,
                    vis_url=to_web_url(vis_path),
                    json_url=to_web_url(json_path),
                    detector=detector,
                    dets=dets,
                )
                add_archive_entry(
                    build_archive_entry_for_detect(
                        job_id,
                        image_url=image_url,
                        vis_url=to_web_url(vis_path),
                        json_url=to_web_url(json_path),
                        detector=detector,
                        num_dets=len(dets),
                        report_url=report_url,
                    )
                )
                json_response(
                    self,
                    {
                        "ok": True,
                        "job_id": job_id,
                        "input_url": image_url,
                        "vis_url": to_web_url(vis_path),
                        "json_url": to_web_url(json_path),
                        "detections": dets,
                        "detector": detector,
                        "report_url": report_url,
                    },
                )
                return
            if parsed.path == "/api/detect_compare":
                payload = read_json_request(self)
                raw_image_url = payload.get("raw_image_url", "")
                enhanced_image_url = payload.get("enhanced_image_url", "")
                detector_key = payload.get("detector_key", "default_detector")
                conf = float(payload.get("conf", 0.25))
                imgsz = int(payload.get("imgsz", 640))
                if not raw_image_url or not enhanced_image_url:
                    error_response(self, "缺少 `raw_image_url` 或 `enhanced_image_url`。")
                    return
                raw_path = web_url_to_path(raw_image_url)
                enhanced_path = web_url_to_path(enhanced_image_url)
                if not raw_path.exists() or not enhanced_path.exists():
                    error_response(self, "`raw/enhanced image_url` 对应文件不存在。")
                    return
                detector = pick_detector(detector_key)
                job_id = f"detect_compare_{uuid.uuid4().hex[:10]}"
                job_root = RUNTIME_ROOT / "detect_compare_jobs" / job_id
                job_root.mkdir(parents=True, exist_ok=True)

                raw_vis_path = job_root / "raw_det_vis.png"
                enh_vis_path = job_root / "enh_det_vis.png"
                raw_json_path = job_root / "raw_detections.json"
                enh_json_path = job_root / "enhanced_detections.json"

                raw_dets = run_detector_on_image(
                    input_path=raw_path,
                    output_vis_path=raw_vis_path,
                    output_json_path=raw_json_path,
                    detector=detector,
                    conf=conf,
                    imgsz=imgsz,
                )
                enhanced_dets = run_detector_on_image(
                    input_path=enhanced_path,
                    output_vis_path=enh_vis_path,
                    output_json_path=enh_json_path,
                    detector=detector,
                    conf=conf,
                    imgsz=imgsz,
                )

                compare_vis_path = job_root / "detect_vs.jpg"
                build_detection_vs_image(raw_vis_path=raw_vis_path, enhanced_vis_path=enh_vis_path, out_path=compare_vis_path)

                raw_vis_url = to_web_url(raw_vis_path)
                enh_vis_url = to_web_url(enh_vis_path)
                compare_vis_url = to_web_url(compare_vis_path)
                raw_json_url = to_web_url(raw_json_path)
                enh_json_url = to_web_url(enh_json_path)
                report_url = build_detect_compare_report(
                    job_root,
                    raw_image_url=raw_image_url,
                    enhanced_image_url=enhanced_image_url,
                    raw_vis_url=raw_vis_url,
                    enhanced_vis_url=enh_vis_url,
                    compare_vis_url=compare_vis_url,
                    raw_json_url=raw_json_url,
                    enhanced_json_url=enh_json_url,
                    detector=detector,
                    raw_dets=raw_dets,
                    enhanced_dets=enhanced_dets,
                )
                add_archive_entry(
                    build_archive_entry_for_detect_compare(
                        job_id,
                        raw_image_url=raw_image_url,
                        enhanced_image_url=enhanced_image_url,
                        compare_vis_url=compare_vis_url,
                        raw_json_url=raw_json_url,
                        enhanced_json_url=enh_json_url,
                        detector=detector,
                        raw_num_dets=len(raw_dets),
                        enhanced_num_dets=len(enhanced_dets),
                        report_url=report_url,
                    )
                )
                json_response(
                    self,
                    {
                        "ok": True,
                        "job_id": job_id,
                        "raw_image_url": raw_image_url,
                        "enhanced_image_url": enhanced_image_url,
                        "raw_vis_url": raw_vis_url,
                        "enhanced_vis_url": enh_vis_url,
                        "compare_vis_url": compare_vis_url,
                        "raw_json_url": raw_json_url,
                        "enhanced_json_url": enh_json_url,
                        "raw_detections": raw_dets,
                        "enhanced_detections": enhanced_dets,
                        "detector": detector,
                        "report_url": report_url,
                    },
                )
                return
            if parsed.path == "/api/compare":
                payload = read_json_request(self)
                filename = payload.get("filename", "upload.png")
                image_data = payload.get("image", "")
                traditional_key = payload.get("traditional_key", "grayworld_clahe")
                improved_key = payload.get("improved_key", "improved_mpcgan")
                if not image_data:
                    error_response(self, "未收到上传图片。")
                    return
                staging_root = RUNTIME_ROOT / "incoming"
                input_path = decode_image_payload(filename, image_data, staging_root)
                result = build_compare_job(input_path, traditional_key, improved_key, self.gpu_ids)
                add_archive_entry(build_archive_entry_for_compare(result["job_id"], result))
                json_response(self, {"ok": True, "result": result})
                return
            error_response(self, "未找到接口。", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args):
        sys.stdout.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))


def make_handler(gpu_ids: str):
    def handler(*args, **kwargs):
        return DemoRequestHandler(*args, gpu_ids=gpu_ids, **kwargs)

    return handler


def main() -> int:
    args = parse_args()
    ensure_runtime_root()
    WEB_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.gpu_ids))
    print(f"Interactive underwater demo: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


