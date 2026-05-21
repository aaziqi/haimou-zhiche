from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ModelView:
    title: str
    pred_dir: Path


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _find_ref(ref_dir: Path, stem: str) -> Path | None:
    cands = sorted([p for p in ref_dir.glob(f"{stem}.*") if p.is_file()])
    if cands:
        return cands[0]
    cands = sorted([p for p in ref_dir.glob(f"{stem}*") if p.is_file()])
    if cands:
        return cands[0]
    return None


def _find_fake(pred_dir: Path, inp_stem: str) -> Path | None:
    candidates = [
        pred_dir / f"{inp_stem}_fake.png",
        pred_dir / f"{inp_stem}_fake.jpg",
        pred_dir / f"{inp_stem}__fake.png",
        pred_dir / f"{inp_stem}__fake.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    cands = sorted(pred_dir.glob(f"{inp_stem}*fake*.png"))
    if cands:
        return cands[0]
    cands = sorted(pred_dir.glob(f"{inp_stem}*fake*.jpg"))
    if cands:
        return cands[0]
    return None


def _setup_sci_style(base_fontsize: int = 9):
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": base_fontsize - 1,
            "ytick.labelsize": base_fontsize - 1,
            "legend.fontsize": base_fontsize - 1,
            "axes.linewidth": 0.8,
        }
    )


def _parse_markdown_table(md_text: str, header_prefix: str):
    lines = md_text.splitlines()
    header_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header_prefix):
            header_line_idx = i
            break
    if header_line_idx is None:
        raise SystemExit(f"failed to find table header starting with: {header_prefix}")

    header = [x.strip() for x in lines[header_line_idx].split("|") if x.strip()]
    rows = []
    for line in lines[header_line_idx + 1:]:
        s = line.strip()
        if not s.startswith("|"):
            break
        parts = [x.strip() for x in s.split("|") if x.strip()]
        if not parts:
            continue
        if all(set(x) <= {"-", ":"} for x in parts):
            continue
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    if not rows:
        raise SystemExit("no rows parsed from markdown table")
    return rows


def _safe_float(x: str) -> float | None:
    s = x.strip()
    if s == "" or s == "—" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def make_ablation_bar_figure(
    euvp_md_path: Path,
    out_dir: Path,
    out_stem: str,
):
    md_text = euvp_md_path.read_text(encoding="utf-8")
    rows = _parse_markdown_table(md_text, header_prefix="| 配置 ID")
    rows = [r for r in rows if r.get("配置 ID") in {"A", "B", "C", "D", "E"}]
    rows.sort(key=lambda r: r["配置 ID"])
    if len(rows) == 0:
        raise SystemExit("table 3 rows A-E not found")

    labels = [r["配置 ID"] for r in rows]
    psnr = [_safe_float(r.get("PSNR_mean", "")) for r in rows]
    ssim = [_safe_float(r.get("SSIM_mean", "")) for r in rows]
    uciqe = [_safe_float(r.get("UCIQE_pred", "")) for r in rows]
    uiqm = [_safe_float(r.get("UIQM_pred", "")) for r in rows]

    metrics = [
        ("PSNR (dB)", psnr),
        ("SSIM", ssim),
        ("UCIQE", uciqe),
        ("UIQM", uiqm),
    ]

    _setup_sci_style(base_fontsize=9)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    x = np.arange(len(labels))

    for i, (title, values) in enumerate(metrics):
        ax = axes[i]
        vals = [v if v is not None else np.nan for v in values]
        bars = ax.bar(x, vals, color=colors[: len(labels)], width=0.72, edgecolor="#222222", linewidth=0.6)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ymax = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else 1.0
        ax.set_ylim(0, ymax * 1.18)
        for b, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + ymax * 0.02,
                f"{v:.3f}" if title == "SSIM" else f"{v:.2f}",
                ha="center",
                va="bottom",
            )
        best_idx = int(np.nanargmax(vals)) if np.any(np.isfinite(vals)) else None
        if best_idx is not None and np.isfinite(vals[best_idx]):
            ax.text(
                x[best_idx],
                vals[best_idx] + ymax * 0.08,
                "best",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111111",
            )
        ax.text(0.02, 0.98, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="left", va="top")

    fig.text(
        0.5,
        0.01,
        "Ablation on EUVP test_samples (matched=200). Models trained on UIEB_Unpaired (20 epochs, max_dataset_size=200).",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_qualitative_grid(
    inp_dir: Path,
    ref_dir: Path,
    views: list[ModelView],
    out_dir: Path,
    out_stem: str,
    nrows: int,
    stems: list[str] | None = None,
    figure_note: str = "Qualitative comparison on EUVP test_samples (A→B).",
):
    inp_files = sorted([p for p in inp_dir.glob("*.jpg") if p.is_file()])
    if not inp_files:
        inp_files = sorted([p for p in inp_dir.glob("*.*") if p.is_file()])
    if not inp_files:
        raise SystemExit(f"no input images found in: {inp_dir}")

    chosen = []
    if stems is not None and len(stems) > 0:
        for stem in stems:
            ref_path = _find_ref(ref_dir, stem)
            if ref_path is None:
                continue
            ok = True
            for v in views:
                if _find_fake(v.pred_dir, stem) is None:
                    ok = False
                    break
            if not ok:
                continue
            chosen.append(stem)
        if len(chosen) == 0:
            raise SystemExit("no valid stems matched for qualitative grid")
    else:
        for p in inp_files:
            stem = p.stem
            ref_path = _find_ref(ref_dir, stem)
            if ref_path is None:
                continue
            ok = True
            for v in views:
                if _find_fake(v.pred_dir, stem) is None:
                    ok = False
                    break
            if not ok:
                continue
            chosen.append(stem)
            if len(chosen) >= nrows:
                break
        if len(chosen) < max(1, nrows):
            raise SystemExit(f"not enough matched cases found, only {len(chosen)}")

    col_titles = ["Input", "Reference"] + [v.title for v in views]
    ncols = len(col_titles)

    _setup_sci_style(base_fontsize=8)
    fig_w = 1.65 * ncols
    fig_h = 1.40 * len(chosen)
    fig, axes = plt.subplots(len(chosen), ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, stem in enumerate(chosen):
        inp_path = inp_dir / f"{stem}.jpg"
        ref_path = _find_ref(ref_dir, stem)
        inp_img = _read_rgb(inp_path)
        ref_img = _read_rgb(ref_path)

        imgs = [inp_img, ref_img]
        for v in views:
            fake_path = _find_fake(v.pred_dir, stem)
            imgs.append(_read_rgb(fake_path))

        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#222222")
            if r == 0:
                ax.set_title(col_titles[c], pad=4)
            if c == 0:
                ax.text(
                    0.01,
                    0.99,
                    stem.replace("test_", ""),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#111111", linewidth=0.6, alpha=0.85),
                )

    fig.text(
        0.5,
        0.01,
        figure_note,
        ha="center",
        va="bottom",
        fontsize=8,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


LOSS_LINE_RE = re.compile(
    r"\(epoch:\s*(?P<epoch>\d+),\s*iters:\s*(?P<iters>\d+).*?"
    r"D_A:\s*(?P<D_A>[-+]?\d*\.?\d+),\s*G_A:\s*(?P<G_A>[-+]?\d*\.?\d+),\s*cycle_A:\s*(?P<cycle_A>[-+]?\d*\.?\d+),\s*idt_A:\s*(?P<idt_A>[-+]?\d*\.?\d+),\s*"
    r"D_B:\s*(?P<D_B>[-+]?\d*\.?\d+),\s*G_B:\s*(?P<G_B>[-+]?\d*\.?\d+),\s*cycle_B:\s*(?P<cycle_B>[-+]?\d*\.?\d+),\s*idt_B:\s*(?P<idt_B>[-+]?\d*\.?\d+)"
)


def _moving_average(xs: list[float], w: int) -> list[float]:
    if w <= 1:
        return xs
    out = []
    s = 0.0
    q: list[float] = []
    for x in xs:
        q.append(float(x))
        s += float(x)
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _parse_loss_log(loss_log: Path):
    records = []
    for line in loss_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LOSS_LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        records.append(
            {
                "epoch": int(d["epoch"]),
                "iters": int(d["iters"]),
                "D": 0.5 * (float(d["D_A"]) + float(d["D_B"])),
                "G_gan": 0.5 * (float(d["G_A"]) + float(d["G_B"])),
                "cycle": 0.5 * (float(d["cycle_A"]) + float(d["cycle_B"])),
                "idt": 0.5 * (float(d["idt_A"]) + float(d["idt_B"])),
            }
        )
    if not records:
        raise SystemExit(f"no loss records parsed from: {loss_log}")
    return records


def _downsample(records, max_points: int):
    if max_points <= 0 or len(records) <= max_points:
        return records
    step = max(1, len(records) // max_points)
    return records[::step]


def make_training_curves_compare(
    series: list[tuple[str, Path]],
    out_dir: Path,
    out_stem: str,
    smooth: int = 25,
    max_points: int = 2500,
):
    _setup_sci_style(base_fontsize=9)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)

    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]
    kinds = [
        ("D", "Discriminator loss"),
        ("G_gan", "Generator GAN loss"),
        ("cycle", "Cycle consistency loss"),
        ("idt", "Identity loss"),
    ]

    for si, (label, loss_log) in enumerate(series):
        recs = _downsample(_parse_loss_log(loss_log), max_points=max_points)
        x = list(range(len(recs)))
        for ax, (k, title) in zip(axes, kinds):
            y = _moving_average([float(r[k]) for r in recs], smooth)
            ax.plot(x, y, linewidth=1.0, label=label, color=palette[si % len(palette)])

    for i, (ax, (_, title)) in enumerate(zip(axes, kinds)):
        ax.set_title(title)
        ax.set_xlabel("Step (log order)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.98, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="left", va="top")
        ax.legend(frameon=False)

    fig.text(
        0.5,
        0.01,
        f"Training curves from loss_log (moving average window={smooth}).",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_training_epoch_means(
    series: list[tuple[str, Path]],
    out_dir: Path,
    out_stem: str,
):
    _setup_sci_style(base_fontsize=9)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)

    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]
    kinds = [
        ("D", "Discriminator loss (epoch mean)"),
        ("G_gan", "Generator GAN loss (epoch mean)"),
        ("cycle", "Cycle consistency loss (epoch mean)"),
        ("idt", "Identity loss (epoch mean)"),
    ]

    for si, (label, loss_log) in enumerate(series):
        recs = _parse_loss_log(loss_log)
        by_epoch: dict[int, dict[str, list[float]]] = {}
        for r in recs:
            e = int(r["epoch"])
            by_epoch.setdefault(e, {"D": [], "G_gan": [], "cycle": [], "idt": []})
            for k in ["D", "G_gan", "cycle", "idt"]:
                by_epoch[e][k].append(float(r[k]))
        epochs = sorted(by_epoch.keys())
        for ax, (k, title) in zip(axes, kinds):
            y = [float(np.mean(by_epoch[e][k])) for e in epochs]
            ax.plot(epochs, y, linewidth=1.2, marker="o", markersize=3.2, label=label, color=palette[si % len(palette)])

    for i, (ax, (_, title)) in enumerate(zip(axes, kinds)):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.98, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="left", va="top")
        ax.legend(frameon=False)

    fig.text(
        0.5,
        0.01,
        "Epoch-averaged training curves (from loss_log).",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _psnr_between(pred_path: Path, ref_path: Path) -> float | None:
    pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
    ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if pred is None or ref is None:
        return None
    if pred.shape[:2] != ref.shape[:2]:
        pred = cv2.resize(pred, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return float(cv2.PSNR(ref, pred))


def select_best_worst_cases_by_psnr(pred_dir: Path, ref_dir: Path, k: int):
    pairs = []
    for fake_path in sorted(pred_dir.glob("test_p*__fake.png")):
        stem = fake_path.name.replace("__fake.png", "").replace("__fake.jpg", "")
        ref_path = _find_ref(ref_dir, stem)
        if ref_path is None:
            continue
        psnr = _psnr_between(fake_path, ref_path)
        if psnr is None:
            continue
        pairs.append((psnr, stem))
    if not pairs:
        raise SystemExit(f"no psnr pairs computed for: {pred_dir}")
    pairs.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(k))
    best = [s for _, s in pairs[:k]]
    worst = [s for _, s in pairs[-k:]]
    return best, worst


def make_epoch_metric_trends(
    euvp_md_path: Path,
    out_dir: Path,
    out_stem: str,
):
    md_text = euvp_md_path.read_text(encoding="utf-8")
    rows = _parse_markdown_table(md_text, header_prefix="| Epoch | PSNR_mean")
    epochs = []
    psnr = []
    ssim = []
    uciqe = []
    uiqm = []
    for r in rows:
        e = _safe_float(r.get("Epoch", ""))
        if e is None:
            continue
        epochs.append(int(e))
        psnr.append(_safe_float(r.get("PSNR_mean", "")) or np.nan)
        ssim.append(_safe_float(r.get("SSIM_mean", "")) or np.nan)
        uciqe.append(_safe_float(r.get("UCIQE_pred", "")) or np.nan)
        uiqm.append(_safe_float(r.get("UIQM_pred", "")) or np.nan)

    if len(epochs) == 0:
        raise SystemExit("no epoch metric rows parsed")

    order = np.argsort(np.asarray(epochs))
    epochs = [epochs[i] for i in order]
    psnr = [psnr[i] for i in order]
    ssim = [ssim[i] for i in order]
    uciqe = [uciqe[i] for i in order]
    uiqm = [uiqm[i] for i in order]

    _setup_sci_style(base_fontsize=9)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)
    series = [
        ("PSNR (dB)", psnr),
        ("SSIM", ssim),
        ("UCIQE (Pred)", uciqe),
        ("UIQM (Pred)", uiqm),
    ]
    for i, (title, ys) in enumerate(series):
        ax = axes[i]
        ax.plot(epochs, ys, linewidth=1.3, marker="o", markersize=4, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        best_idx = int(np.nanargmax(np.asarray(ys))) if np.any(np.isfinite(ys)) else None
        if best_idx is not None and np.isfinite(ys[best_idx]):
            ax.scatter([epochs[best_idx]], [ys[best_idx]], color="#E45756", s=22, zorder=3)
            ax.text(epochs[best_idx], ys[best_idx], " best", ha="left", va="center", fontsize=8)
        ax.text(0.02, 0.98, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="left", va="top")

    fig.text(
        0.5,
        0.01,
        "Metric trends of CycleGAN baseline on EUVP test_samples (matched=200).",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_methods_bar_figure(
    euvp_md_path: Path,
    out_dir: Path,
    out_stem: str,
):
    md_text = euvp_md_path.read_text(encoding="utf-8")
    rows = _parse_markdown_table(md_text, header_prefix="| 方法")
    if len(rows) == 0:
        raise SystemExit("no method rows parsed")

    labels = [r.get("方法", "") for r in rows]
    psnr = [_safe_float(r.get("PSNR_mean", "")) for r in rows]
    ssim = [_safe_float(r.get("SSIM_mean", "")) for r in rows]
    uciqe = [_safe_float(r.get("UCIQE_pred", "")) for r in rows]
    uiqm = [_safe_float(r.get("UIQM_pred", "")) for r in rows]

    metrics = [
        ("PSNR (dB)", psnr),
        ("SSIM", ssim),
        ("UCIQE (Pred)", uciqe),
        ("UIQM (Pred)", uiqm),
    ]

    _setup_sci_style(base_fontsize=9)
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2), constrained_layout=True)
    axes = axes.reshape(-1)
    x = np.arange(len(labels))
    colors = ["#4C78A8"] * len(labels)

    for i, (title, values) in enumerate(metrics):
        ax = axes[i]
        vals = [v if v is not None else np.nan for v in values]
        bars = ax.bar(x, vals, color=colors, width=0.72, edgecolor="#222222", linewidth=0.6)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ymax = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else 1.0
        ax.set_ylim(0, ymax * 1.18)
        best_idx = int(np.nanargmax(vals)) if np.any(np.isfinite(vals)) else None
        if best_idx is not None and np.isfinite(vals[best_idx]):
            bars[best_idx].set_color("#E45756")
            ax.text(
                x[best_idx],
                vals[best_idx] + ymax * 0.06,
                "best",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.text(0.02, 0.98, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="left", va="top")

    fig.text(
        0.5,
        0.01,
        "Method comparison on EUVP test_samples (matched=200).",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/paper_figures",
    )
    parser.add_argument(
        "--euvp_md",
        type=str,
        default="docs/EUVP.md",
    )
    parser.add_argument(
        "--inp_dir",
        type=str,
        default=r"d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp",
    )
    parser.add_argument(
        "--gtr_dir",
        type=str,
        default=r"d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr",
    )
    parser.add_argument("--nrows", type=int, default=6)
    parser.add_argument("--best_worst_k", type=int, default=6)
    parser.add_argument("--smooth", type=int, default=25)
    args = parser.parse_args()

    root = Path(".")
    out_dir = root / args.out_dir

    views = [
        ModelView(
            title="CycleGAN (EUVP, e=200)",
            pred_dir=root / "results" / "euvp_cyclegan_full" / "test_200" / "images",
        ),
        ModelView(
            title="Ablation A (EUVP, latest)",
            pred_dir=root / "results" / "euvp_stage2_A_s0" / "test_latest" / "images",
        ),
        ModelView(
            title="Ablation D (EUVP, e=50)",
            pred_dir=root / "results" / "euvp_abl_D_s0" / "test_50" / "images",
        ),
    ]

    q_png, q_pdf = make_qualitative_grid(
        inp_dir=Path(args.inp_dir),
        ref_dir=Path(args.gtr_dir),
        views=views,
        out_dir=out_dir,
        out_stem="fig_qualitative_euvp",
        nrows=int(args.nrows),
    )
    print("Saved:", q_png)
    print("Saved:", q_pdf)

    b_png, b_pdf = make_ablation_bar_figure(
        euvp_md_path=root / args.euvp_md,
        out_dir=out_dir,
        out_stem="fig_ablation_metrics_table3",
    )
    print("Saved:", b_png)
    print("Saved:", b_pdf)

    ablation_views = [
        ModelView(title="A", pred_dir=root / "results" / "abl_uieb_A" / "test_20" / "images"),
        ModelView(title="B", pred_dir=root / "results" / "abl_uieb_B" / "test_20" / "images"),
        ModelView(title="C", pred_dir=root / "results" / "abl_uieb_C" / "test_20" / "images"),
        ModelView(title="D", pred_dir=root / "results" / "abl_uieb_D" / "test_20" / "images"),
        ModelView(title="E", pred_dir=root / "results" / "abl_uieb_E" / "test_20" / "images"),
    ]
    qa_png, qa_pdf = make_qualitative_grid(
        inp_dir=Path(args.inp_dir),
        ref_dir=Path(args.gtr_dir),
        views=ablation_views,
        out_dir=out_dir,
        out_stem="fig_qualitative_ablation_AE",
        nrows=min(int(args.nrows), 4),
        figure_note="Qualitative ablation (A–E) on EUVP test_samples. Models trained on UIEB_Unpaired (20 epochs).",
    )
    print("Saved:", qa_png)
    print("Saved:", qa_pdf)

    epoch_views = [
        ModelView(title="e=150", pred_dir=root / "results" / "euvp_cyclegan_full" / "test_150" / "images"),
        ModelView(title="e=170", pred_dir=root / "results" / "euvp_cyclegan_full" / "test_170" / "images"),
        ModelView(title="e=190", pred_dir=root / "results" / "euvp_cyclegan_full" / "test_190" / "images"),
        ModelView(title="e=200", pred_dir=root / "results" / "euvp_cyclegan_full" / "test_200" / "images"),
    ]
    epq_png, epq_pdf = make_qualitative_grid(
        inp_dir=Path(args.inp_dir),
        ref_dir=Path(args.gtr_dir),
        views=epoch_views,
        out_dir=out_dir,
        out_stem="fig_qualitative_epoch_progress_150_170_190_200",
        nrows=min(int(args.nrows), 4),
        figure_note="Qualitative training progress of CycleGAN on EUVP (epochs 150/170/190/200).",
    )
    print("Saved:", epq_png)
    print("Saved:", epq_pdf)

    baseline_pred_dir = root / "results" / "euvp_cyclegan_full" / "test_200" / "images"
    best, worst = select_best_worst_cases_by_psnr(baseline_pred_dir, Path(args.gtr_dir), k=int(args.best_worst_k))
    bw_views = [ModelView(title="CycleGAN (e=200)", pred_dir=baseline_pred_dir)]
    bw_png, bw_pdf = make_qualitative_grid(
        inp_dir=Path(args.inp_dir),
        ref_dir=Path(args.gtr_dir),
        views=bw_views,
        out_dir=out_dir,
        out_stem="fig_best_worst_cases_cyclegan_e200",
        nrows=len(best) + len(worst),
        stems=best + worst,
        figure_note=f"Best-{len(best)} and worst-{len(worst)} cases by PSNR (CycleGAN e=200).",
    )
    print("Saved:", bw_png)
    print("Saved:", bw_pdf)

    loss_series = [
        ("CycleGAN (EUVP full)", root / "checkpoints" / "euvp_cyclegan_full" / "loss_log.txt"),
        ("Ablation A (EUVP)", root / "checkpoints" / "euvp_stage2_A_s0" / "loss_log.txt"),
        ("Ablation D (EUVP)", root / "checkpoints" / "euvp_abl_D_s0" / "loss_log.txt"),
    ]
    lc_png, lc_pdf = make_training_curves_compare(
        series=loss_series,
        out_dir=out_dir,
        out_stem="fig_training_curves_compare",
        smooth=int(args.smooth),
    )
    print("Saved:", lc_png)
    print("Saved:", lc_pdf)

    lce_png, lce_pdf = make_training_epoch_means(
        series=loss_series,
        out_dir=out_dir,
        out_stem="fig_training_epoch_means_compare",
    )
    print("Saved:", lce_png)
    print("Saved:", lce_pdf)

    mt_png, mt_pdf = make_methods_bar_figure(
        euvp_md_path=root / args.euvp_md,
        out_dir=out_dir,
        out_stem="fig_methods_comparison_table1",
    )
    print("Saved:", mt_png)
    print("Saved:", mt_pdf)

    tr_png, tr_pdf = make_epoch_metric_trends(
        euvp_md_path=root / args.euvp_md,
        out_dir=out_dir,
        out_stem="fig_baseline_epoch_metric_trends",
    )
    print("Saved:", tr_png)
    print("Saved:", tr_pdf)


if __name__ == "__main__":
    main()
