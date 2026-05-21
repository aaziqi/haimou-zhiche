from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PY = REPO_ROOT / "test.py"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="待强的单张图片或文件夹")
    parser.add_argument("--reference_dir", default="", help="叉参考真值目录，用于计算 PSNR/SSIM")
    parser.add_argument("--checkpoint_name", default="euvp_cyclegan_full")
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--direction", default="AtoB", choices=["AtoB", "BtoA"])
    parser.add_argument("--gpu_ids", default="-1")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "competition_demo_outputs" / "national_award_showcase"))
    parser.add_argument("--report_title", default="面向计算机计大赛的水下图像增强演示系统")
    parser.add_argument("--track_name", default="软件应用与开发")
    parser.add_argument("--num_test", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_metric_functions():
    script_path = REPO_ROOT / "scripts" / "evaluate_euvp_psnr_ssim.py"
    spec = importlib.util.spec_from_file_location("euvp_metrics", script_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_key(stem: str) -> str:
    s = stem.strip()
    lower = s.lower()
    for token in [
        "__fake",
        "_fake",
        "-fake",
        " fake",
        "__real",
        "_real",
        "-real",
        " real",
        "__gtr",
        "_gtr",
        "-gtr",
        " gtr",
        "__gt",
        "_gt",
        "-gt",
        " gt",
        "__reference",
        "_reference",
        "-reference",
        " reference",
        "__ref",
        "_ref",
        "-ref",
        " ref",
        "__target",
        "_target",
        "-target",
        " target",
    ]:
        idx = lower.find(token)
        if idx != -1:
            s = s[:idx]
            lower = s.lower()
    for prefix in ["raw_", "input_", "inp_", "underwater_", "uw_", "ref_", "gt_", "gtr_"]:
        if lower.startswith(prefix):
            s = s[len(prefix):].strip()
            lower = s.lower()
    return s.rstrip("_- ").strip()


def index_images(directory: Path) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    if not directory.exists():
        return indexed
    for path in sorted(x for x in directory.glob("**/*") if is_image_file(x)):
        key = normalize_key(path.stem)
        indexed.setdefault(key, []).append(path)
    return indexed


def pick_prediction(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    ranked: list[tuple[int, str, Path]] = []
    for candidate in candidates:
        name = candidate.name.lower()
        score = 0
        if "fake" in name:
            score -= 20
        if "enhance" in name or "result" in name:
            score -= 10
        if "real" in name:
            score += 20
        ranked.append((score, name, candidate))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def detect_model_suffix(checkpoint_dir: Path, epoch: str, direction: str) -> str | None:
    has_a = (checkpoint_dir / f"{epoch}_net_G_A.pth").exists()
    has_b = (checkpoint_dir / f"{epoch}_net_G_B.pth").exists()
    has_g = (checkpoint_dir / f"{epoch}_net_G.pth").exists()
    if direction == "BtoA":
        if has_b:
            return "_B"
        if has_a:
            return "_A"
        if has_g:
            return ""
        return None
    if has_a:
        return "_A"
    if has_b:
        return "_B"
    if has_g:
        return ""
    return None


def parse_train_opt(checkpoint_dir: Path) -> dict[str, str]:
    train_opt = checkpoint_dir / "train_opt.txt"
    options: dict[str, str] = {}
    if not train_opt.exists():
        return options
    for raw_line in train_opt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in raw_line or raw_line.startswith("---"):
            continue
        key, value = raw_line.split(":", 1)
        clean_value = value.split("\t")[0].strip()
        options[key.strip()] = clean_value
    return options


def prepare_input_directory(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_dir():
        return input_path, None
    if not is_image_file(input_path):
        raise FileNotFoundError(f"输入跾不是可创? {input_path}")
    temp_dir = tempfile.TemporaryDirectory(prefix="competition_demo_input_")
    temp_root = Path(temp_dir.name)
    shutil.copy2(input_path, temp_root / input_path.name)
    return temp_root, temp_dir


def count_images(directory: Path) -> int:
    return sum(1 for path in directory.glob("**/*") if is_image_file(path))


def build_test_command(
    input_dir: Path,
    output_dir: Path,
    checkpoint_name: str,
    epoch: str,
    direction: str,
    num_test: int,
) -> tuple[list[str], Path]:
    checkpoint_dir = CHECKPOINTS_DIR / checkpoint_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"有?checkpoint 盽: {checkpoint_dir}")
    suffix = detect_model_suffix(checkpoint_dir, epoch, direction)
    if suffix is None:
        raise FileNotFoundError(f"有到可用生成器权重: {checkpoint_dir}")
    options = parse_train_opt(checkpoint_dir)
    command = [
        sys.executable,
        str(TEST_PY),
        "--dataroot",
        str(input_dir),
        "--name",
        checkpoint_name,
        "--model",
        "test",
        "--dataset_mode",
        "single",
        "--results_dir",
        str(output_dir / "inference_raw"),
        "--epoch",
        epoch,
        "--num_test",
        str(num_test),
    ]
    passthrough_keys = [
        "netG",
        "norm",
        "ngf",
        "input_nc",
        "output_nc",
        "load_size",
        "crop_size",
        "preprocess",
    ]
    for key in passthrough_keys:
        value = options.get(key)
        if value:
            command.extend([f"--{key}", value])
    if options.get("no_dropout", "").lower() == "true":
        command.append("--no_dropout")
    if suffix:
        command.extend(["--model_suffix", suffix])
    web_dir = output_dir / "inference_raw" / checkpoint_name / f"test_{epoch}"
    return command, web_dir


def run_inference(command: list[str], gpu_ids: str) -> None:
    env = dict(**__import__("os").environ)
    if gpu_ids.strip() == "-1":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["CYCLEGAN_DEVICE"] = "cpu"
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{completed.stdout}\n{completed.stderr}".strip())


def relative_copy(source: Path, destination_dir: Path, name: str) -> str:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / name
    shutil.copy2(source, target)
    return target.name


@dataclass
class ImageRecord:
    key: str
    input_file: str
    output_file: str
    reference_file: str
    input_uciqe: float | None
    output_uciqe: float | None
    delta_uciqe: float | None
    input_uiqm: float | None
    output_uiqm: float | None
    delta_uiqm: float | None
    input_psnr: float | None
    output_psnr: float | None
    delta_psnr: float | None
    input_ssim: float | None
    output_ssim: float | None
    delta_ssim: float | None


def safe_round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def average(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def resize_like(module, image, target):
    height, width = target.shape[:2]
    if image.shape[:2] == (height, width):
        return image
    interpolation = module.INTER_AREA if image.shape[0] > height or image.shape[1] > width else module.INTER_CUBIC
    return module.resize(image, (width, height), interpolation=interpolation)


def compute_record(
    metrics_module,
    key: str,
    input_path: Path,
    output_path: Path,
    reference_path: Path | None,
    assets_dir: Path,
) -> ImageRecord:
    input_asset = relative_copy(input_path, assets_dir / "input", input_path.name)
    output_asset = relative_copy(output_path, assets_dir / "output", output_path.name)
    reference_asset = ""
    input_uciqe = output_uciqe = delta_uciqe = None
    input_uiqm = output_uiqm = delta_uiqm = None
    input_psnr = output_psnr = delta_psnr = None
    input_ssim = output_ssim = delta_ssim = None
    if metrics_module is not None:
        input_bgr = metrics_module.read_image(input_path)
        output_bgr = metrics_module.read_image(output_path)
        input_uciqe = metrics_module.compute_uciqe(input_bgr)
        output_uciqe = metrics_module.compute_uciqe(output_bgr)
        input_uiqm = metrics_module.compute_uiqm(input_bgr)
        output_uiqm = metrics_module.compute_uiqm(output_bgr)
        delta_uciqe = output_uciqe - input_uciqe
        delta_uiqm = output_uiqm - input_uiqm
        if reference_path is not None and reference_path.exists():
            reference_asset = relative_copy(reference_path, assets_dir / "reference", reference_path.name)
            reference_bgr = metrics_module.read_image(reference_path)
            aligned_input = resize_like(metrics_module.cv2, input_bgr, reference_bgr)
            aligned_output = resize_like(metrics_module.cv2, output_bgr, reference_bgr)
            input_psnr = metrics_module.compute_psnr(aligned_input, reference_bgr)
            output_psnr = metrics_module.compute_psnr(aligned_output, reference_bgr)
            input_ssim = metrics_module.compute_ssim(aligned_input, reference_bgr)
            output_ssim = metrics_module.compute_ssim(aligned_output, reference_bgr)
            delta_psnr = output_psnr - input_psnr
            delta_ssim = output_ssim - input_ssim
    elif reference_path is not None and reference_path.exists():
        reference_asset = relative_copy(reference_path, assets_dir / "reference", reference_path.name)
    return ImageRecord(
        key=key,
        input_file=f"assets/input/{input_asset}",
        output_file=f"assets/output/{output_asset}",
        reference_file=f"assets/reference/{reference_asset}" if reference_asset else "",
        input_uciqe=safe_round(input_uciqe),
        output_uciqe=safe_round(output_uciqe),
        delta_uciqe=safe_round(delta_uciqe),
        input_uiqm=safe_round(input_uiqm),
        output_uiqm=safe_round(output_uiqm),
        delta_uiqm=safe_round(delta_uiqm),
        input_psnr=safe_round(input_psnr),
        output_psnr=safe_round(output_psnr),
        delta_psnr=safe_round(delta_psnr),
        input_ssim=safe_round(input_ssim),
        output_ssim=safe_round(output_ssim),
        delta_ssim=safe_round(delta_ssim),
    )


def build_records(
    input_dir: Path,
    prediction_dir: Path,
    reference_dir: Path | None,
    assets_dir: Path,
) -> list[ImageRecord]:
    metrics_module = load_metric_functions()
    indexed_inputs = index_images(input_dir)
    indexed_predictions = index_images(prediction_dir)
    indexed_references = index_images(reference_dir) if reference_dir is not None else {}
    records: list[ImageRecord] = []
    for key in sorted(indexed_inputs.keys()):
        input_candidates = indexed_inputs.get(key, [])
        prediction_candidates = indexed_predictions.get(key, [])
        if not input_candidates or not prediction_candidates:
            continue
        input_path = input_candidates[0]
        output_path = pick_prediction(prediction_candidates)
        if output_path is None:
            continue
        reference_candidates = indexed_references.get(key, [])
        reference_path = reference_candidates[0] if reference_candidates else None
        records.append(compute_record(metrics_module, key, input_path, output_path, reference_path, assets_dir))
    return records


def write_metrics_csv(records: list[ImageRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(records[0]).keys()) if records else ["key"])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def metric_card(title: str, value: float | None, suffix: str = "") -> str:
    if value is None:
        display = "?"
    else:
        sign = "+" if value > 0 else ""
        display = f"{sign}{value:.4f}{suffix}"
    return f"<div class='metric-card'><div class='metric-title'>{html.escape(title)}</div><div class='metric-value'>{display}</div></div>"


def record_metric_line(label: str, before: float | None, after: float | None, delta: float | None) -> str:
    if before is None or after is None:
        return ""
    delta_text = ""
    if delta is not None:
        sign = "+" if delta > 0 else ""
        delta_text = f"（变化 {sign}{delta:.4f}）"
    return f"<div class='row-metric'><span>{html.escape(label)}</span><span>{before:.4f} -> {after:.4f} {delta_text}</span></div>"


def build_html_report(
    records: list[ImageRecord],
    output_dir: Path,
    title: str,
    track_name: str,
    checkpoint_name: str,
    epoch: str,
    raw_result_dir: Path,
) -> None:
    summary = {
        "image_count": len(records),
        "avg_delta_uciqe": safe_round(average([record.delta_uciqe for record in records])),
        "avg_delta_uiqm": safe_round(average([record.delta_uiqm for record in records])),
        "avg_delta_psnr": safe_round(average([record.delta_psnr for record in records])),
        "avg_delta_ssim": safe_round(average([record.delta_ssim for record in records])),
    }
    cards = [
        metric_card("平均 UCIQE 提升", summary["avg_delta_uciqe"]),
        metric_card("平均 UIQM 提升", summary["avg_delta_uiqm"]),
        metric_card("平均 PSNR 提升", summary["avg_delta_psnr"]),
        metric_card("平均 SSIM 提升", summary["avg_delta_ssim"]),
    ]
    sections = []
    for index, record in enumerate(records, start=1):
        reference_block = (
            f"<div class='image-panel'><div class='image-title'>参图?/div><img src='{html.escape(record.reference_file)}' alt='reference'></div>"
            if record.reference_file
            else ""
        )
        metrics_html = "".join(
            [
                record_metric_line("UCIQE", record.input_uciqe, record.output_uciqe, record.delta_uciqe),
                record_metric_line("UIQM", record.input_uiqm, record.output_uiqm, record.delta_uiqm),
                record_metric_line("PSNR", record.input_psnr, record.output_psnr, record.delta_psnr),
                record_metric_line("SSIM", record.input_ssim, record.output_ssim, record.delta_ssim),
            ]
        )
        sections.append(
            f"""
            <section class='case-card'>
                <div class='case-header'>
                    <div class='case-index'>样例 {index:02d}</div>
                    <div class='case-key'>{html.escape(record.key)}</div>
                </div>
                <div class='image-grid'>
                    <div class='image-panel'><div class='image-title'>原水下图像</div><img src='{html.escape(record.input_file)}' alt='input'></div>
                    <div class='image-panel'><div class='image-title'>增强结果</div><img src='{html.escape(record.output_file)}' alt='output'></div>
                    {reference_block}
                </div>
                <div class='metric-list'>{metrics_html or "<div class='row-metric'><span>指标说明</span><span>当前朐用或朏供参考图?/span></div>"}</div>
            </section>
            """
        )
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{html.escape(title)}</title>
        <style>
            :root {{
                --bg: #07111f;
                --panel: rgba(13, 29, 52, 0.88);
                --panel-soft: rgba(22, 42, 74, 0.72);
                --text: #f4f8ff;
                --muted: #9cb4d3;
                --accent: #4bc3ff;
                --accent-soft: rgba(75, 195, 255, 0.16);
                --border: rgba(120, 180, 255, 0.18);
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
                background: radial-gradient(circle at top, #133763 0%, #07111f 55%, #050b15 100%);
                color: var(--text);
            }}
            .wrap {{ width: min(1200px, calc(100% - 40px)); margin: 0 auto; padding: 36px 0 56px; }}
            .hero {{
                background: linear-gradient(135deg, rgba(28, 81, 142, 0.92), rgba(10, 23, 46, 0.96));
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 32px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
            }}
            .hero h1 {{ margin: 0 0 14px; font-size: 34px; line-height: 1.3; }}
            .hero p {{ margin: 0; color: var(--muted); font-size: 15px; line-height: 1.8; }}
            .hero-tags {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px; }}
            .tag {{
                padding: 8px 14px;
                border-radius: 999px;
                background: var(--accent-soft);
                color: #dff5ff;
                border: 1px solid rgba(75, 195, 255, 0.28);
                font-size: 14px;
            }}
            .metrics {{
                margin-top: 24px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
                gap: 16px;
            }}
            .metric-card {{
                background: var(--panel-soft);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 18px;
            }}
            .metric-title {{ color: var(--muted); font-size: 14px; margin-bottom: 12px; }}
            .metric-value {{ font-size: 28px; font-weight: 700; color: #ffffff; }}
            .section-title {{ margin: 34px 0 16px; font-size: 24px; }}
            .section-desc {{ margin: 0 0 22px; color: var(--muted); line-height: 1.8; }}
            .case-card {{
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 24px;
                margin-bottom: 22px;
                box-shadow: 0 18px 48px rgba(0, 0, 0, 0.2);
            }}
            .case-header {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 12px;
                margin-bottom: 18px;
            }}
            .case-index {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 88px;
                height: 36px;
                padding: 0 14px;
                border-radius: 999px;
                background: var(--accent-soft);
                color: #dff5ff;
                font-weight: 700;
            }}
            .case-key {{ color: var(--muted); font-size: 14px; }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 18px;
            }}
            .image-panel {{
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 18px;
                padding: 14px;
            }}
            .image-title {{ margin-bottom: 12px; color: var(--muted); font-size: 14px; }}
            .image-panel img {{
                width: 100%;
                height: 260px;
                object-fit: contain;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 14px;
            }}
            .metric-list {{
                margin-top: 18px;
                display: grid;
                gap: 10px;
            }}
            .row-metric {{
                display: flex;
                justify-content: space-between;
                gap: 16px;
                padding: 12px 14px;
                border-radius: 14px;
                background: rgba(255, 255, 255, 0.03);
                color: #dbe8ff;
                font-size: 14px;
            }}
            .footer {{
                margin-top: 28px;
                color: var(--muted);
                font-size: 13px;
                line-height: 1.8;
            }}
            @media (max-width: 720px) {{
                .wrap {{ width: min(100% - 24px, 1200px); }}
                .hero {{ padding: 22px; }}
                .hero h1 {{ font-size: 28px; }}
                .row-metric {{ flex-direction: column; }}
            }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <section class="hero">
                <h1>{html.escape(title)}</h1>
                <p>本页面面向软件应用展示场景，围绕水下图像增强模型的可视化演示、指标对比与样例输出进行统一包装，可直接用于作品展示与成果归档。</p>
                <div class="hero-tags">
                    <div class="tag">参加赛道：{html.escape(track_name)}</div>
                    <div class="tag">模型权重：{html.escape(checkpoint_name)}</div>
                    <div class="tag">权重版本：{html.escape(epoch)}</div>
                    <div class="tag">样例数量：{len(records)}</div>
                </div>
                <div class="metrics">{''.join(cards)}</div>
            </section>
            <h2 class="section-title">系统说明</h2>
            <p class="section-desc">当前页面基于现有 CycleGAN 水下增强主框架，无需改动训练主干即可完成单图或批量增强、指标统计和展示页生成，适合作为软件应用展示层与成果说明页。</p>
            <h2 class="section-title">增强案例</h2>
            <p class="section-desc">每个样例同时展示原图像、增强结果以及可选参考图像，并给出 UCIQE、UIQM、PSNR、SSIM 等指标变化。</p>
            {''.join(sections)}
            <div class="footer">
                <div>报告生成时间：{generated_time}</div>
                <div>原推理结果盽：{html.escape(str(raw_result_dir))}</div>
                <div>你可以直接打当前 index.html 进线下展示，或配合朜静服务进行浏览器演示?/div>
            </div>
        </div>
    </body>
    </html>
    """
    (output_dir / "index.html").write_text(report_html, encoding="utf-8")
    summary_json = {
        "report_title": title,
        "track_name": track_name,
        "checkpoint_name": checkpoint_name,
        "epoch": epoch,
        "generated_at": generated_time,
        "raw_result_dir": str(raw_result_dir),
        "summary": summary,
        "records": [asdict(record) for record in records],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir, args.overwrite)
    input_dir, temp_dir = prepare_input_directory(input_path)
    try:
        num_test = args.num_test if args.num_test > 0 else count_images(input_dir)
        command, raw_result_dir = build_test_command(
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint_name=args.checkpoint_name,
            epoch=args.epoch,
            direction=args.direction,
            num_test=num_test,
        )
        run_inference(command, args.gpu_ids)
        prediction_dir = raw_result_dir / "images"
        if not prediction_dir.exists():
            raise FileNotFoundError(f"没有找到推理输出目录: {prediction_dir}")
        assets_dir = output_dir / "assets"
        records = build_records(input_dir, prediction_dir, reference_dir, assets_dir)
        if not records:
            raise RuntimeError("没有找到可匹配的输入图像与增强结果，请检查输入目录和输出目录。")
        write_metrics_csv(records, output_dir / "metrics.csv")
        build_html_report(
            records=records,
            output_dir=output_dir,
            title=args.report_title,
            track_name=args.track_name,
            checkpoint_name=args.checkpoint_name,
            epoch=args.epoch,
            raw_result_dir=raw_result_dir,
        )
        print(f"展示报告已生成：{output_dir / 'index.html'}")
        print(f"指标汇文件：{output_dir / 'metrics.csv'}")
        print(f"结构化摘要：{output_dir / 'summary.json'}")
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

