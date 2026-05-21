import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
OUT_DIR = PROJECT_ROOT / "ppt_visuals"

LOSS_MPCGAN = REPO_ROOT / "checkpoints" / "euvp_mpcgan_stage2_s0" / "loss_log.txt"
LOSS_CYCLEGAN = REPO_ROOT / "checkpoints" / "euvp_cyclegan_full" / "loss_log.txt"
SUMMARY_CSV = REPO_ROOT / "results" / "benchmarks" / "summary.csv"
SETTING_A = PROJECT_ROOT / "results" / "downstream_det_settingA_fixed_mpc_full" / "downstream_detection_results.json"
SETTING_B = PROJECT_ROOT / "results" / "downstream_det_settingB_enh_mpc_full" / "downstream_detection_results.json"
ARCHIVE_JSON = REPO_ROOT / "local_web_demo" / "runtime" / "archive" / "entries.json"

FIG_DPI = 180


def ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_matplotlib():
    plt.style.use("dark_background")
    plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.facecolor"] = "#0d1b2a"
    plt.rcParams["figure.facecolor"] = "#07101a"
    plt.rcParams["savefig.facecolor"] = "#07101a"
    plt.rcParams["axes.edgecolor"] = "#33506b"
    plt.rcParams["axes.labelcolor"] = "#dcecff"
    plt.rcParams["xtick.color"] = "#dcecff"
    plt.rcParams["ytick.color"] = "#dcecff"
    plt.rcParams["text.color"] = "#f4fbff"
    plt.rcParams["grid.color"] = "#2f4358"


def parse_loss_log(path: Path):
    pattern = re.compile(
        r"\(epoch:\s*(?P<epoch>\d+),\s*iters:\s*(?P<iters>\d+),.*?\)\s*,\s*(?P<metrics>.+)$"
    )
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        item = {
            "epoch": int(m.group("epoch")),
            "iters": int(m.group("iters")),
        }
        for part in m.group("metrics").split(","):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            try:
                item[k.strip()] = float(v.strip())
            except ValueError:
                pass
        records.append(item)
    return records


def smooth(values, window=5):
    if len(values) <= window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_loss_overview(records, title, metrics, out_name):
    xs = [r["epoch"] + r["iters"] / 10000.0 for r in records]
    fig, axes = plt.subplots(2, math.ceil(len(metrics) / 2), figsize=(14, 7), dpi=FIG_DPI)
    axes = axes.flatten()
    colors = ["#53b3ff", "#44d7b6", "#ffb454", "#ff6b9a", "#7c8cff", "#9be564"]
    for idx, metric in enumerate(metrics):
        ys = [r.get(metric, float("nan")) for r in records]
        ys = [y for y in ys if not math.isnan(y)]
        if not ys:
            continue
        full_y = [r.get(metric, float("nan")) for r in records]
        valid_x = [x for x, y in zip(xs, full_y) if not math.isnan(y)]
        valid_y = [y for y in full_y if not math.isnan(y)]
        ax = axes[idx]
        ax.plot(valid_x, valid_y, color=colors[idx % len(colors)], alpha=0.35, linewidth=1.2)
        ax.plot(valid_x, smooth(valid_y, window=4), color=colors[idx % len(colors)], linewidth=2.4)
        ax.set_title(metric)
        ax.grid(True, alpha=0.35)
    for ax in axes[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def read_summary_rows():
    rows = []
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def pick_row(rows, dataset, model, epoch=None):
    matches = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
    if epoch is not None:
        matches = [r for r in matches if r["epoch"] == str(epoch)]
    return matches[0] if matches else None


def to_float(row, key):
    try:
        return float(row[key]) if row and row.get(key, "") not in ("", None) else float("nan")
    except Exception:
        return float("nan")


def plot_benchmark_group(rows, dataset, models, keys, out_name, title):
    labels = [m[0] for m in models]
    width = 0.16
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIG_DPI)
    palette = ["#4cc9f0", "#2dd4bf", "#ffd166", "#ff7b72"]
    for idx, key in enumerate(keys):
        vals = []
        for _, model_name, epoch in models:
            row = pick_row(rows, dataset, model_name, epoch)
            vals.append(to_float(row, key))
        ax.bar([i + (idx - 1.5) * width for i in x], vals, width=width, label=key, color=palette[idx])
        for xi, yi in zip([i + (idx - 1.5) * width for i in x], vals):
            if not math.isnan(yi):
                ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_downstream(setting_a, setting_b):
    metrics = [
        ("Precision", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)"),
        ("mAP50", "metrics/mAP50(B)"),
        ("mAP50-95", "metrics/mAP50-95(B)"),
    ]
    for title, payload, out_name in [
        ("Setting A: Fixed Detector", setting_a, "05_downstream_settingA.png"),
        ("Setting B: Enhancement-aware", setting_b, "06_downstream_settingB.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
        x = range(len(metrics))
        raw = [payload["metrics_raw"][k] for _, k in metrics]
        enh = [payload["metrics_enhanced"][k] for _, k in metrics]
        ax.bar([i - 0.18 for i in x], raw, width=0.35, label="Raw", color="#667eea")
        ax.bar([i + 0.18 for i in x], enh, width=0.35, label="Enhanced", color="#2dd4bf")
        ax.set_xticks(list(x))
        ax.set_xticklabels([m[0] for m in metrics])
        ax.set_ylim(0, max(raw + enh) * 1.22)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        for xi, yi in zip([i - 0.18 for i in x], raw):
            ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
        for xi, yi in zip([i + 0.18 for i in x], enh):
            ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    deltas = [
        setting_a["metrics_enhanced"][k] - setting_a["metrics_raw"][k]
        for _, k in metrics
    ]
    deltas_b = [
        setting_b["metrics_enhanced"][k] - setting_b["metrics_raw"][k]
        for _, k in metrics
    ]
    x = range(len(metrics))
    ax.bar([i - 0.18 for i in x], deltas, width=0.35, label="Setting A delta", color="#ff7b72")
    ax.bar([i + 0.18 for i in x], deltas_b, width=0.35, label="Setting B delta", color="#4cc9f0")
    ax.axhline(0, color="#dcecff", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_title("Detection Delta: Enhanced - Raw", fontsize=16, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_downstream_delta_compare.png", bbox_inches="tight")
    plt.close(fig)


def resolve_web_url(url: str):
    return REPO_ROOT / "local_web_demo" / url.lstrip("/").replace("/", "\\")


def get_font(size=22):
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_enhancement_board(entries):
    chosen = [e for e in entries if e["type"] == "enhance"][:3]
    if not chosen:
        return
    width, row_h, pad, title_h = 1600, 360, 24, 90
    board = Image.new("RGB", (width, title_h + len(chosen) * (row_h + pad) + pad), (7, 16, 26))
    draw = ImageDraw.Draw(board)
    title_font = get_font(34)
    text_font = get_font(22)
    draw.text((pad, 24), "Enhancement Cases for PPT", fill=(240, 248, 255), font=title_font)
    for idx, item in enumerate(chosen):
        y = title_h + idx * (row_h + pad)
        raw = Image.open(resolve_web_url(item["input_url"])).convert("RGB")
        enh = Image.open(resolve_web_url(item["cover_url"])).convert("RGB")
        raw = raw.resize((520, 300))
        enh = enh.resize((520, 300))
        board.paste(raw, (pad, y + 30))
        board.paste(enh, (pad + 560, y + 30))
        draw.text((pad, y), f"Case {idx + 1} - Raw", fill=(210, 230, 255), font=text_font)
        draw.text((pad + 560, y), f"Case {idx + 1} - Enhanced", fill=(210, 230, 255), font=text_font)
        draw.text((pad + 1120, y + 56), item["summary"], fill=(240, 248, 255), font=text_font)
        draw.text((pad + 1120, y + 108), item["model_label"], fill=(140, 201, 255), font=text_font)
    board.save(OUT_DIR / "08_enhancement_case_board.png")


def build_detection_board(entries):
    chosen = [e for e in entries if e["type"] == "detect_compare"][:3]
    if not chosen:
        return
    width, row_h, pad, title_h = 1600, 310, 24, 90
    board = Image.new("RGB", (width, title_h + len(chosen) * (row_h + pad) + pad), (7, 16, 26))
    draw = ImageDraw.Draw(board)
    title_font = get_font(34)
    text_font = get_font(22)
    draw.text((pad, 24), "Detection Comparison Cases", fill=(240, 248, 255), font=title_font)
    for idx, item in enumerate(chosen):
        y = title_h + idx * (row_h + pad)
        det = Image.open(resolve_web_url(item["cover_url"])).convert("RGB").resize((980, 240))
        board.paste(det, (pad, y + 36))
        draw.text((pad, y), f"Case {idx + 1}", fill=(210, 230, 255), font=text_font)
        draw.text((1028, y + 52), item["summary"], fill=(240, 248, 255), font=text_font)
        draw.text((1028, y + 98), "Meaning: show whether enhancement helps detection.", fill=(140, 201, 255), font=text_font)
    board.save(OUT_DIR / "09_detection_case_board.png")


def write_manifest():
    text = """PPT visuals generated from existing training/evaluation artifacts.

01_training_loss_cyclegan_full.png
- Full CycleGAN training loss overview
- Suitable for: training process / baseline stability

02_training_loss_mpcgan_stage2.png
- MP-CycleGAN stage2 fine-tuning losses
- Suitable for: improved model training details

03_euvp_main_benchmark.png
- Main method comparison on EUVP (PSNR/SSIM/UCIQE/UIQM)
- Suitable for: in-domain quality comparison

04_uieb_main_benchmark.png
- Main method comparison on UIEB
- Suitable for: generalization comparison

05_downstream_settingA.png
06_downstream_settingB.png
07_downstream_delta_compare.png
- Downstream detection support figures
- Suitable for: task-oriented validation

08_enhancement_case_board.png
- 3 representative raw vs enhanced cases
- Suitable for: qualitative visual results

09_detection_case_board.png
- 3 representative raw vs enhanced detection comparison boards
- Suitable for: direct response to "does enhancement help tasks?"
"""
    (OUT_DIR / "README.txt").write_text(text, encoding="utf-8")


def main():
    ensure_out()
    configure_matplotlib()

    cyc_records = parse_loss_log(LOSS_CYCLEGAN)
    mpc_records = parse_loss_log(LOSS_MPCGAN)

    plot_loss_overview(
        cyc_records,
        "CycleGAN Training Loss Overview",
        ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B"],
        "01_training_loss_cyclegan_full.png",
    )
    plot_loss_overview(
        mpc_records,
        "MP-CycleGAN Stage2 Loss Overview",
        ["D_A", "G_A", "cycle_A", "idt_A", "struct_A", "perc_A"],
        "02_training_loss_mpcgan_stage2.png",
    )

    rows = read_summary_rows()
    main_models = [
        ("Identity", "baseline:identity", None),
        ("GrayWorld", "baseline:grayworld", None),
        ("CLAHE", "baseline:clahe", None),
        ("CycleGAN", "euvp_cyclegan_full", 200),
        ("MP-CycleGAN", "euvp_mpcgan_stage2_s0", 202),
    ]
    plot_benchmark_group(
        rows,
        "euvp",
        main_models,
        ["psnr", "ssim", "uciqe_pred", "uiqm_pred"],
        "03_euvp_main_benchmark.png",
        "EUVP Main Benchmark",
    )
    plot_benchmark_group(
        rows,
        "uieb",
        main_models,
        ["psnr", "ssim", "uciqe_pred", "uiqm_pred"],
        "04_uieb_main_benchmark.png",
        "UIEB Generalization Benchmark",
    )

    setting_a = load_json(SETTING_A)
    setting_b = load_json(SETTING_B)
    plot_downstream(setting_a, setting_b)

    entries = json.loads(ARCHIVE_JSON.read_text(encoding="utf-8"))
    build_enhancement_board(entries)
    build_detection_board(entries)
    write_manifest()
    print(f"Generated PPT visuals in: {OUT_DIR}")


if __name__ == "__main__":
    main()
