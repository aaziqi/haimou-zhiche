import os
import shutil
import random
import glob
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

# Config
NUM_IMAGES = 20
OUTPUT_DIR = r"UserStudy_Package"
SOURCE_INP_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\Inp"
SOURCE_OURS_DIR = r"results\euvp_stage2_A_s0\test_latest\images"  # Contains _fake.png
SOURCE_BASE_DIR = r"results\euvp_cyclegan_full\test_200\images"  # Contains _fake.png
REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_user_study():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Get list of test images
    all_inputs = glob.glob(os.path.join(SOURCE_INP_DIR, "*.jpg"))
    if not all_inputs:
        all_inputs = glob.glob(os.path.join(SOURCE_INP_DIR, "*.png"))

    selected = random.sample(all_inputs, min(len(all_inputs), NUM_IMAGES))

    html_content = """
    <html>
    <head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .comparison { margin-bottom: 40px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }
        .images { display: flex; justify-content: space-around; }
        .img-container { text-align: center; }
        img { max-width: 300px; max-height: 300px; }
        h3 { color: #333; }
    </style>
    </head>
    <body>
    <h1>Underwater Image Enhancement User Study</h1>
    <p>Please rate each method from 1 (Worst) to 5 (Best) based on color naturalness and detail clarity.</p>
    <form>
    """

    for i, inp_path in enumerate(selected):
        basename = os.path.basename(inp_path)
        name_no_ext = os.path.splitext(basename)[0]

        # Paths for methods
        # Note: CycleGAN results usually append _fake.png
        # Check standard naming
        ours_name = name_no_ext + "_fake.png"
        if not os.path.exists(os.path.join(SOURCE_OURS_DIR, ours_name)):
            ours_name = name_no_ext + "__fake.png"

        base_name = name_no_ext + "_fake.png"

        # Copy images to package
        img_dir = os.path.join(OUTPUT_DIR, "images")
        os.makedirs(img_dir, exist_ok=True)

        dest_inp = os.path.join(img_dir, f"{i}_input.png")
        dest_ours = os.path.join(img_dir, f"{i}_methodA.png")  # Blind naming
        dest_base = os.path.join(img_dir, f"{i}_methodB.png")  # Blind naming

        # Copy Input
        # Convert jpg to png if needed for consistency or just copy
        shutil.copy2(inp_path, dest_inp)

        # Copy Ours
        src_ours = os.path.join(SOURCE_OURS_DIR, ours_name)
        if os.path.exists(src_ours):
            shutil.copy2(src_ours, dest_ours)

        # Copy Baseline
        src_base = os.path.join(SOURCE_BASE_DIR, base_name)
        if os.path.exists(src_base):
            shutil.copy2(src_base, dest_base)
        else:
            # Fallback if baseline missing (create dummy or skip)
            pass

        # Randomize A/B display order in HTML to avoid bias
        methods = [
            {"id": "A", "src": f"images/{i}_methodA.png", "real_name": "Ours"},
            {"id": "B", "src": f"images/{i}_methodB.png", "real_name": "Baseline"}
        ]
        random.shuffle(methods)

        html_content += f"""
        <div class="comparison">
            <h3>Scene {i+1}</h3>
            <div class="images">
                <div class="img-container">
                    <img src="images/{i}_input.png">
                    <p>Input</p>
                </div>
                <div class="img-container">
                    <img src="{methods[0]['src']}">
                    <p>Method 1</p>
                    <label>Score: <input type="number" min="1" max="5" name="s{i}_m1"></label>
                </div>
                <div class="img-container">
                    <img src="{methods[1]['src']}">
                    <p>Method 2</p>
                    <label>Score: <input type="number" min="1" max="5" name="s{i}_m2"></label>
                </div>
            </div>
        </div>
        """

    html_content += """
    <button type="button" onclick="alert('Thank you! Please save this page as PDF or screenshot and send it back.')">Submit</button>
    </form>
    </body>
    </html>
    """

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
        f.write(html_content)

    print(f"User Study package created at {os.path.abspath(OUTPUT_DIR)}")
    print("Instructions: Zip this folder and send to 10-15 friends. Ask them to open index.html and rate the images.")


def _read_csv_by_key(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            k = (r.get("key") or "").strip()
            if not k:
                continue
            rows[k] = r
    return rows


def _safe_float(x: str | None) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _imread_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_letterbox(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    th, tw = size
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _resize_cover_crop(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    th, tw = size
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = max(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x0 = max(0, (nw - tw) // 2)
    y0 = max(0, (nh - th) // 2)
    crop = resized[y0:y0 + th, x0:x0 + tw]
    if crop.shape[0] != th or crop.shape[1] != tw:
        crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    return crop


def _put_label(img: np.ndarray, text: str | list[str]) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    lines = [text] if isinstance(text, str) else list(text)
    pad = 6
    x0, y0 = 0, 0
    line_gap = 4
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    tw = max([s[0] for s in sizes], default=0)
    th = max([s[1] for s in sizes], default=0)
    box_h = pad * 2 + len(lines) * th + max(0, len(lines) - 1) * line_gap
    box_w = pad * 2 + tw
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    y = y0 + pad + th
    for t in lines:
        cv2.putText(out, t, (x0 + pad, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += th + line_gap
    return out


def _resolve_any_path(s: str) -> Path:
    p = Path(s)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _fmt_delta(x: float) -> str:
    return f"+{x:.2f}" if x >= 0 else f"{x:.2f}"


def _hstack(images: list[np.ndarray], sep: int = 10, sep_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    if not images:
        raise ValueError("empty images")
    h = images[0].shape[0]
    for im in images[1:]:
        if im.shape[0] != h:
            raise ValueError("height mismatch")
    if sep <= 0:
        return np.concatenate(images, axis=1)
    bar = np.full((h, sep, 3), sep_color, dtype=np.uint8)
    out = images[0]
    for im in images[1:]:
        out = np.concatenate([out, bar, im], axis=1)
    return out


def _vstack(images: list[np.ndarray], sep: int = 10, sep_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    if not images:
        raise ValueError("empty images")
    w = images[0].shape[1]
    for im in images[1:]:
        if im.shape[1] != w:
            raise ValueError("width mismatch")
    if sep <= 0:
        return np.concatenate(images, axis=0)
    bar = np.full((sep, w, 3), sep_color, dtype=np.uint8)
    out = images[0]
    for im in images[1:]:
        out = np.concatenate([out, bar, im], axis=0)
    return out


def _set_ieee_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "figure.dpi": 600,
            "savefig.dpi": 600,
        }
    )


def _save_ieee_grid(
    rows: list[list[np.ndarray]],
    col_titles: list[str],
    row_titles: list[str] | None,
    out_path: Path,
    two_col: bool = True,
):
    _set_ieee_style()
    nrows = len(rows)
    ncols = len(col_titles)
    if nrows == 0 or ncols == 0:
        raise ValueError("empty grid")
    w_in = 7.16 if two_col else 3.5
    wspace_in = 0.06
    hspace_in = 0.08
    margin_t_in = 0.18
    margin_b_in = 0.12
    margin_l_in = 0.10 if row_titles else 0.06
    margin_r_in = 0.06

    cell_w_in = (w_in - margin_l_in - margin_r_in - (ncols - 1) * wspace_in) / ncols
    cell_h_in = cell_w_in
    h_in = margin_t_in + margin_b_in + nrows * cell_h_in + (nrows - 1) * hspace_in

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w_in, h_in), squeeze=False)
    plt.subplots_adjust(
        left=margin_l_in / w_in,
        right=1 - margin_r_in / w_in,
        top=1 - margin_t_in / h_in,
        bottom=margin_b_in / h_in,
        wspace=wspace_in / cell_w_in,
        hspace=hspace_in / cell_h_in,
    )

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.imshow(rows[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.3)
                spine.set_edgecolor((0, 0, 0, 0.35))
            if r == 0:
                ax.set_title(col_titles[c], pad=2)
            if c == 0 and row_titles:
                ax.set_ylabel(row_titles[r], rotation=0, ha="right", va="center", labelpad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def make_uieb_cross_domain_grid(
    cyc_per_image_csv: Path,
    mp_per_image_csv: Path,
    out_path: Path,
    num_pairs: int = 6,
    pick_metric: str = "uiqm_pred",
    cell_hw: tuple[int, int] = (360, 360),
):
    cyc = _read_csv_by_key(cyc_per_image_csv)
    mp = _read_csv_by_key(mp_per_image_csv)
    keys = sorted(set(cyc.keys()) & set(mp.keys()))
    if not keys:
        raise SystemExit("No overlapping keys between the two per_image.csv files.")

    scored = []
    for k in keys:
        a = cyc.get(k, {})
        b = mp.get(k, {})
        va = _safe_float(a.get(pick_metric))
        vb = _safe_float(b.get(pick_metric))
        if va is None or vb is None:
            continue
        scored.append((k, vb - va))
    if not scored:
        raise SystemExit(f"No valid numeric values found for metric={pick_metric}.")

    scored.sort(key=lambda x: x[1], reverse=True)
    k_pos = max(1, num_pairs // 2)
    k_neg = max(1, num_pairs - k_pos)
    top = [k for k, _ in scored[:k_pos * 5]]
    bot = [k for k, _ in scored[-k_neg * 5:]][::-1]

    selected: list[str] = []
    for k in top:
        if k not in selected:
            selected.append(k)
        if len(selected) >= k_pos:
            break
    for k in bot:
        if k not in selected:
            selected.append(k)
        if len(selected) >= num_pairs:
            break

    th, tw = cell_hw
    rows: list[list[np.ndarray]] = []
    row_titles: list[str] = []
    for k in selected:
        a = cyc[k]
        b = mp[k]

        inp_path_s = (b.get("inp_path") or a.get("inp_path") or "").strip()
        ref_path_s = (b.get("ref_path") or a.get("ref_path") or "").strip()
        cyc_pred_s = (a.get("pred_path") or "").strip()
        mp_pred_s = (b.get("pred_path") or "").strip()

        inp_path = _resolve_any_path(inp_path_s) if inp_path_s else Path()
        ref_path = _resolve_any_path(ref_path_s) if ref_path_s else Path()
        cyc_pred = _resolve_any_path(cyc_pred_s) if cyc_pred_s else Path()
        mp_pred = _resolve_any_path(mp_pred_s) if mp_pred_s else Path()

        inp = _imread_bgr(inp_path)
        ref = _imread_bgr(ref_path)
        cyc_img = _imread_bgr(cyc_pred)
        mp_img = _imread_bgr(mp_pred)
        if inp is None or ref is None or cyc_img is None or mp_img is None:
            continue

        uiqm_c = _safe_float(a.get("uiqm_pred"))
        uiqm_m = _safe_float(b.get("uiqm_pred"))
        d_uiqm = (uiqm_m - uiqm_c) if uiqm_m is not None and uiqm_c is not None else None
        title = f"ΔUIQM={_fmt_delta(d_uiqm)}" if d_uiqm is not None else ""

        inp = _bgr_to_rgb(_resize_cover_crop(inp, (th, tw)))
        cyc_img = _bgr_to_rgb(_resize_cover_crop(cyc_img, (th, tw)))
        mp_img = _bgr_to_rgb(_resize_cover_crop(mp_img, (th, tw)))
        ref = _bgr_to_rgb(_resize_cover_crop(ref, (th, tw)))

        rows.append([inp, cyc_img, mp_img, ref])
        row_titles.append(title)
        if len(rows) >= num_pairs:
            break

    if not rows:
        raise SystemExit("No valid image rows could be built from selected keys.")
    _save_ieee_grid(
        rows=rows,
        col_titles=["(a) Input", "(b) CycleGAN", "(c) MP-CycleGAN", "(d) GT"],
        row_titles=row_titles,
        out_path=out_path,
        two_col=True,
    )


def make_euvp_visual_grid(
    cyc_per_image_csv: Path,
    mp_per_image_csv: Path,
    out_path: Path,
    num_cases: int = 4,
    cell_hw: tuple[int, int] = (640, 640),
):
    cyc = _read_csv_by_key(cyc_per_image_csv)
    mp = _read_csv_by_key(mp_per_image_csv)
    keys = sorted(set(cyc.keys()) & set(mp.keys()))
    if not keys:
        raise SystemExit("No overlapping keys between the two per_image.csv files.")

    scored = []
    for k in keys:
        a = cyc.get(k, {})
        b = mp.get(k, {})
        psnr_c = _safe_float(a.get("psnr"))
        psnr_m = _safe_float(b.get("psnr"))
        ssim_c = _safe_float(a.get("ssim"))
        ssim_m = _safe_float(b.get("ssim"))
        if psnr_c is None or psnr_m is None or ssim_c is None or ssim_m is None:
            continue
        score = (psnr_m - psnr_c) + 20.0 * (ssim_m - ssim_c)
        scored.append((score, k))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for _, k in scored:
        a = cyc.get(k, {})
        b = mp.get(k, {})
        inp_path_s = (b.get("inp_path") or a.get("inp_path") or "").strip()
        ref_path_s = (b.get("ref_path") or a.get("ref_path") or "").strip()
        cyc_pred_s = (a.get("pred_path") or "").strip()
        mp_pred_s = (b.get("pred_path") or "").strip()
        if not inp_path_s or not ref_path_s or not cyc_pred_s or not mp_pred_s:
            continue
        inp = _imread_bgr(_resolve_any_path(inp_path_s))
        ref = _imread_bgr(_resolve_any_path(ref_path_s))
        cyc_img = _imread_bgr(_resolve_any_path(cyc_pred_s))
        mp_img = _imread_bgr(_resolve_any_path(mp_pred_s))
        if inp is None or ref is None or cyc_img is None or mp_img is None:
            continue
        selected.append(k)
        if len(selected) >= int(num_cases):
            break

    if not selected:
        raise SystemExit("No valid cases could be selected to build EUVP grid.")

    th, tw = cell_hw
    rows: list[list[np.ndarray]] = []
    for k in selected:
        a = cyc[k]
        b = mp[k]

        inp = _imread_bgr(_resolve_any_path((b.get("inp_path") or a.get("inp_path") or "").strip()))
        ref = _imread_bgr(_resolve_any_path((b.get("ref_path") or a.get("ref_path") or "").strip()))
        cyc_img = _imread_bgr(_resolve_any_path((a.get("pred_path") or "").strip()))
        mp_img = _imread_bgr(_resolve_any_path((b.get("pred_path") or "").strip()))
        if inp is None or ref is None or cyc_img is None or mp_img is None:
            continue

        inp = _bgr_to_rgb(_resize_cover_crop(inp, (th, tw)))
        cyc_img = _bgr_to_rgb(_resize_cover_crop(cyc_img, (th, tw)))
        mp_img = _bgr_to_rgb(_resize_cover_crop(mp_img, (th, tw)))
        ref = _bgr_to_rgb(_resize_cover_crop(ref, (th, tw)))

        rows.append([inp, cyc_img, mp_img, ref])

    _save_ieee_grid(
        rows=rows,
        col_titles=["(a) Input", "(b) CycleGAN", "(c) MP-CycleGAN", "(d) GT"],
        row_titles=None,
        out_path=out_path,
        two_col=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="user_study", choices=["user_study", "uieb_grid", "euvp_grid"])
    parser.add_argument("--out", default=str(Path("docs") / "figures" / "uieb_cross_domain_visuals.png"))
    parser.add_argument("--num_pairs", type=int, default=6)
    parser.add_argument("--metric", default="uiqm_pred")
    parser.add_argument(
        "--cyc_csv",
        default=str(Path("results") / "benchmarks" / "uieb" / "models" / "euvp_cyclegan_full" / "epoch_200" / "per_image.csv"),
    )
    parser.add_argument(
        "--mp_csv",
        default=str(Path("results") / "benchmarks" / "uieb" / "models" / "euvp_mpcgan_stage2_s0" / "epoch_202" / "per_image.csv"),
    )
    parser.add_argument(
        "--euvp_out",
        default=str(Path("docs") / "figures" / "visual_comparison_final_refined.png"),
    )
    parser.add_argument(
        "--euvp_cyc_csv",
        default=str(Path("results") / "benchmarks" / "euvp" / "models" / "euvp_cyclegan_full" / "epoch_200" / "per_image.csv"),
    )
    parser.add_argument(
        "--euvp_mp_csv",
        default=str(Path("results") / "benchmarks" / "euvp" / "models" / "euvp_mpcgan_stage2_s0" / "epoch_202" / "per_image.csv"),
    )
    parser.add_argument("--euvp_cases", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "user_study":
        setup_user_study()
        return

    if args.mode == "uieb_grid":
        make_uieb_cross_domain_grid(
            cyc_per_image_csv=Path(args.cyc_csv),
            mp_per_image_csv=Path(args.mp_csv),
            out_path=Path(args.out),
            num_pairs=int(args.num_pairs),
            pick_metric=str(args.metric),
        )
        return

    make_euvp_visual_grid(
        cyc_per_image_csv=Path(args.euvp_cyc_csv),
        mp_per_image_csv=Path(args.euvp_mp_csv),
        out_path=Path(args.euvp_out),
        num_cases=int(args.euvp_cases),
    )


if __name__ == "__main__":
    main()
