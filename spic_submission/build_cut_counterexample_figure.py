from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(r"D:\VScode\Graduation project")
REPO_ROOT = (
    PROJECT_ROOT
    / "pytorch-CycleGAN-and-pix2pix-master"
    / "pytorch-CycleGAN-and-pix2pix-master"
)
UNPAIRED_ROOT = PROJECT_ROOT / "Unpaired_Baselines"
OUT_DIR = PROJECT_ROOT / "spic_submission" / "figures_used"

MP_CSV = (
    REPO_ROOT
    / "results"
    / "benchmarks"
    / "euvp"
    / "models"
    / "euvp_mpcgan_continue250_from200_s0"
    / "epoch_250"
    / "per_image.csv"
)
CYCLE_CSV = (
    REPO_ROOT
    / "results"
    / "benchmarks"
    / "euvp"
    / "models"
    / "euvp_cyclegan_continue250_from200_s0"
    / "epoch_250"
    / "per_image.csv"
)
CUT_CSV = PROJECT_ROOT / "spic_submission" / "cut_euvp_per_image.csv"
FASTCUT_CSV = PROJECT_ROOT / "spic_submission" / "fastcut_euvp_per_image.csv"

OUT_PATH = OUT_DIR / "Fig_cut_fastcut_counterexample_spic.png"
MANIFEST_PATH = OUT_DIR / "cut_counterexample_manifest.json"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    subtitle: str
    zoom_boxes: tuple[tuple[float, float, float, float], tuple[float, float, float, float]]
    cut_notes: tuple[str, str]
    mp_notes: tuple[str, str]


CASE_SPECS: tuple[CaseSpec, ...] = (
    CaseSpec(
        key="test_p280",
        title="Case 1",
        subtitle="Artificial-like elongated edge and junction topology",
        zoom_boxes=((0.18, 0.14, 0.22, 0.22), (0.62, 0.36, 0.20, 0.20)),
        cut_notes=("halo and edge warp", "junction distortion"),
        mp_notes=("clean contour", "stable junction"),
    ),
    CaseSpec(
        key="test_p206",
        title="Case 2",
        subtitle="Dense coral texture and fine-detail stability",
        zoom_boxes=((0.44, 0.23, 0.20, 0.20), (0.18, 0.58, 0.20, 0.20)),
        cut_notes=("blotchy texture", "false speckle"),
        mp_notes=("natural coral", "smooth shading"),
    ),
)


def load_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["key"]: row for row in csv.DictReader(f)}


def _to_float(row: dict[str, str], field: str) -> float:
    return float((row.get(field) or "").strip())


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_cover(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    h, w = img.shape[:2]
    scale = max(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def _crop_patch(img: np.ndarray, box: tuple[float, float, float, float], patch_size: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x = int(round(box[0] * w))
    y = int(round(box[1] * h))
    bw = int(round(box[2] * w))
    bh = int(round(box[3] * h))
    x = max(0, min(x, w - 2))
    y = max(0, min(y, h - 2))
    bw = max(2, min(bw, w - x))
    bh = max(2, min(bh, h - y))
    crop = img[y : y + bh, x : x + bw]
    return _resize_cover(crop, patch_size)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    font_path = Path(r"C:\Windows\Fonts") / name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    fallback = Path(r"C:\Windows\Fonts") / "arial.ttf"
    return ImageFont.truetype(str(fallback), size)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    top_y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((center_x - text_w / 2, top_y), text, font=font, fill=fill)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int = 2,
) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or len(lines) + 1 >= max_lines:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1 :])]
    return lines


def _draw_wrapped_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    top_y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 2,
) -> None:
    lines = _wrap_text(draw, text, font, max_width=max_width, max_lines=2)
    cursor_y = top_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_h = bbox[3] - bbox[1]
        _draw_centered_text(draw, line, center_x, cursor_y, font, fill)
        cursor_y += text_h + line_gap


def _draw_box(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], color: tuple[int, int, int], width: int = 4) -> None:
    for offset in range(width):
        draw.rectangle((rect[0] - offset, rect[1] - offset, rect[2] + offset, rect[3] + offset), outline=color)


def _draw_dashed_box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int = 3,
    dash: int = 10,
    gap: int = 6,
) -> None:
    x0, y0, x1, y1 = rect
    for offset in range(width):
        xa, ya, xb, yb = x0 - offset, y0 - offset, x1 + offset, y1 + offset
        cursor = xa
        while cursor < xb:
            end = min(cursor + dash, xb)
            draw.line((cursor, ya, end, ya), fill=color, width=1)
            draw.line((cursor, yb, end, yb), fill=color, width=1)
            cursor += dash + gap
        cursor = ya
        while cursor < yb:
            end = min(cursor + dash, yb)
            draw.line((xa, cursor, xa, end), fill=color, width=1)
            draw.line((xb, cursor, xb, end), fill=color, width=1)
            cursor += dash + gap


def _build_case_panel(
    case: CaseSpec,
    rows: dict[str, dict[str, dict[str, str]]],
    fonts: dict[str, ImageFont.ImageFont],
) -> tuple[Image.Image, dict[str, float]]:
    label_order = [
        ("(a) Degraded Input", rows["mp"][case.key]["inp_path"]),
        ("(b) CycleGAN", rows["cycle"][case.key]["pred_path"]),
        ("(c) CUT", rows["cut"][case.key]["pred_path"]),
        ("(d) FastCUT", rows["fast"][case.key]["pred_path"]),
        ("(e) MP-CycleGAN", rows["mp"][case.key]["pred_path"]),
        ("(f) Reference", rows["mp"][case.key]["ref_path"]),
    ]
    main_size = (196, 196)
    patch_size = (84, 84)
    margin_x = 24
    gap_x = 14
    patch_gap = 20
    title_h = 60
    stat_h = 34
    header_h = 28
    main_h = main_size[0]
    patch_h = patch_size[0]
    canvas_w = margin_x * 2 + len(label_order) * main_size[1] + (len(label_order) - 1) * gap_x
    note_h = 40
    canvas_h = title_h + stat_h + header_h + main_h + 22 + patch_h + note_h + 38

    panel = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    draw.text((margin_x, 10), case.title, font=fonts["title"], fill=(0, 0, 0))
    draw.text((margin_x, 34), case.subtitle, font=fonts["subtitle"], fill=(70, 70, 70))

    mp_row = rows["mp"][case.key]
    cut_row = rows["cut"][case.key]
    fast_row = rows["fast"][case.key]
    stat_text = (
        "MP-CUT: {:+.2f} dB / {:+.4f} SSIM    "
        "MP-FastCUT: {:+.2f} dB / {:+.4f} SSIM".format(
            _to_float(mp_row, "psnr") - _to_float(cut_row, "psnr"),
            _to_float(mp_row, "ssim") - _to_float(cut_row, "ssim"),
            _to_float(mp_row, "psnr") - _to_float(fast_row, "psnr"),
            _to_float(mp_row, "ssim") - _to_float(fast_row, "ssim"),
        )
    )
    draw.text((margin_x, 62), stat_text, font=fonts["stat"], fill=(40, 40, 40))

    main_y = title_h + stat_h + header_h
    patch_y = main_y + main_h + 22

    rgb_images = {}
    pil_images = {}
    for label, path_str in label_order:
        rgb = _read_rgb(Path(path_str))
        rgb_images[label] = rgb
        pil_images[label] = Image.fromarray(_resize_cover(rgb, main_size))

    for idx, (label, _) in enumerate(label_order):
        x = margin_x + idx * (main_size[1] + gap_x)
        _draw_centered_text(
            draw,
            label,
            x + main_size[1] // 2,
            main_y - 24,
            fonts["header"],
            (0, 0, 0),
        )
        panel.paste(pil_images[label], (x, main_y))
        draw.rectangle((x, main_y, x + main_size[1], main_y + main_size[0]), outline=(0, 0, 0), width=2)

        color = (220, 30, 70)
        for box in case.zoom_boxes:
            bx = x + int(round(box[0] * main_size[1]))
            by = main_y + int(round(box[1] * main_size[0]))
            bw = int(round(box[2] * main_size[1]))
            bh = int(round(box[3] * main_size[0]))
            _draw_dashed_box(draw, (bx, by, bx + bw, by + bh), color, width=2)

        for patch_idx, box in enumerate(case.zoom_boxes):
            patch_group_w = len(case.zoom_boxes) * patch_size[1] + (len(case.zoom_boxes) - 1) * patch_gap
            patch_start_x = x + (main_size[1] - patch_group_w) // 2
            px = patch_start_x + patch_idx * (patch_size[1] + patch_gap)
            patch = Image.fromarray(_crop_patch(rgb_images[label], box, patch_size))
            panel.paste(patch, (px, patch_y))
            draw.rectangle((px, patch_y, px + patch_size[1], patch_y + patch_size[0]), outline=(220, 30, 70), width=3)

            note = None
            note_color = None
            if label in {"(c) CUT", "(d) FastCUT"}:
                note = case.cut_notes[patch_idx]
                note_color = (210, 0, 0)
            elif label == "(e) MP-CycleGAN":
                note = case.mp_notes[patch_idx]
                note_color = (0, 130, 40)
            if note:
                if label in {"(c) CUT", "(d) FastCUT"}:
                    arrow_start = (px + patch_size[1] - 10, patch_y + 12)
                    arrow_end = (px + patch_size[1] - 34, patch_y + 30)
                    draw.line((arrow_start, arrow_end), fill=note_color, width=3)
                    draw.polygon(
                        (
                            arrow_start,
                            (arrow_start[0] - 10, arrow_start[1] + 4),
                            (arrow_start[0] - 3, arrow_start[1] + 12),
                        ),
                        fill=note_color,
                    )
                    _draw_wrapped_centered_text(
                        draw,
                        note,
                        px + patch_size[1] // 2,
                        patch_y + patch_size[0] + 4,
                        fonts["note"],
                        note_color,
                        max_width=patch_size[1] + 12,
                    )
                else:
                    _draw_wrapped_centered_text(
                        draw,
                        "\u2713 " + note,
                        px + patch_size[1] // 2,
                        patch_y + patch_size[0] + 4,
                        fonts["note"],
                        note_color,
                        max_width=patch_size[1] + 12,
                    )

    return panel, {
        "mp_minus_cut_psnr": round(_to_float(mp_row, "psnr") - _to_float(cut_row, "psnr"), 4),
        "mp_minus_cut_ssim": round(_to_float(mp_row, "ssim") - _to_float(cut_row, "ssim"), 6),
        "mp_minus_fastcut_psnr": round(_to_float(mp_row, "psnr") - _to_float(fast_row, "psnr"), 4),
        "mp_minus_fastcut_ssim": round(_to_float(mp_row, "ssim") - _to_float(fast_row, "ssim"), 6),
    }


def main() -> None:
    rows = {
        "mp": load_rows(MP_CSV),
        "cycle": load_rows(CYCLE_CSV),
        "cut": load_rows(CUT_CSV),
        "fast": load_rows(FASTCUT_CSV),
    }
    fonts = {
        "title": _font("timesbd.ttf", 24),
        "subtitle": _font("times.ttf", 17),
        "header": _font("timesbd.ttf", 16),
        "stat": _font("times.ttf", 15),
        "note": _font("timesbd.ttf", 12),
    }

    panels = []
    manifest = {"figure": str(OUT_PATH), "cases": []}
    for case in CASE_SPECS:
        panel, stats = _build_case_panel(case, rows, fonts)
        panels.append(panel)
        entry = {"key": case.key, "title": case.title, "subtitle": case.subtitle}
        entry.update(stats)
        manifest["cases"].append(entry)

    gap_y = 28
    final_w = max(p.width for p in panels)
    final_h = sum(p.height for p in panels) + gap_y * (len(panels) - 1) + 28
    final = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    y = 0
    for idx, panel in enumerate(panels):
        final.paste(panel, (0, y))
        y += panel.height
        if idx != len(panels) - 1:
            y += gap_y

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final.save(OUT_PATH, quality=95, dpi=(300, 300))
    MANIFEST_PATH.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved figure to: {OUT_PATH}")
    print(f"Saved manifest to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
