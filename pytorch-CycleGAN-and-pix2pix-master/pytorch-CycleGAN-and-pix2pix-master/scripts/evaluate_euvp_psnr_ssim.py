r"""
Evaluate PSNR and SSIM for EUVP test_samples.

Assumes directory structure:
- Inp: input images (e.g., d:\...\EUVP Dataset\test_samples\Inp)
- GTr: ground truth reference images
- Pred: generated images by model, filenames matching Inp (or mapped via a simple pattern).

This script matches files by basename (without extension). If your generated outputs from test.py
are named differently (e.g., saved as HTML with labels), you may specify a glob or rename them accordingly.

Usage:
  python scripts/evaluate_euvp_psnr_ssim.py --inp_dir "...\EUVP Dataset\test_samples\Inp" --gtr_dir "...\EUVP Dataset\test_samples\GTr" --pred_dir "...\results\euvp_cyclegan\test_latest\images"

"""

import argparse
from pathlib import Path
import json
import csv
import cv2
import numpy as np


def compute_psnr(img1, img2):
    return cv2.PSNR(img1, img2)


def compute_uciqe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = l_channel * (100.0 / 255.0)
    a_channel = a_channel - 128.0
    b_channel = b_channel - 128.0
    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)
    sigma_c = float(np.std(chroma))
    con_l = float(np.percentile(l_channel, 99) - np.percentile(l_channel, 1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    s_channel = hsv[:, :, 1] / 255.0
    mu_s = float(np.mean(s_channel))
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


def _trimmed_stats(x: np.ndarray, alpha: float = 0.1):
    v = np.sort(x.reshape(-1).astype(np.float64))
    n = int(v.size)
    if n == 0:
        return 0.0, 0.0
    k = int(alpha * n)
    if n - 2 * k <= 0:
        t = v
    else:
        t = v[k:n - k]
    return float(np.mean(t)), float(np.std(t))


def _eme(x: np.ndarray, block_size: int = 8, eps: float = 1e-12) -> float:
    x = x.astype(np.float64)
    h, w = x.shape[:2]
    k1 = h // block_size
    k2 = w // block_size
    if k1 <= 0 or k2 <= 0:
        return 0.0
    x = x[:k1 * block_size, :k2 * block_size]
    s = 0.0
    for i in range(k1):
        for j in range(k2):
            block = x[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            s += float(np.log((bmax + eps) / (bmin + eps)))
    return float((2.0 / (k1 * k2)) * s)


def _compute_uicm(bgr: np.ndarray) -> float:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    mu_rg, sigma_rg = _trimmed_stats(rg)
    mu_yb, sigma_yb = _trimmed_stats(yb)
    return float((-0.0268 * np.sqrt(mu_rg * mu_rg + mu_yb * mu_yb)) + (0.1586 * np.sqrt(sigma_rg * sigma_rg + sigma_yb * sigma_yb)))


def _compute_uism(bgr: np.ndarray) -> float:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    emes = []
    for c in range(3):
        ch = rgb[:, :, c]
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        emes.append(_eme(mag))
    return float(0.299 * emes[0] + 0.587 * emes[1] + 0.114 * emes[2])


def _compute_uiconm(bgr: np.ndarray, block_size: int = 8, eps: float = 1e-12) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape[:2]
    k1 = h // block_size
    k2 = w // block_size
    if k1 <= 0 or k2 <= 0:
        return 0.0
    gray = gray[:k1 * block_size, :k2 * block_size]
    s = 0.0
    for i in range(k1):
        for j in range(k2):
            block = gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            s += (bmax - bmin) / (bmax + bmin + eps)
    return float(s / (k1 * k2))


def compute_uiqm(image):
    uicm = _compute_uicm(image)
    uism = _compute_uism(image)
    uiconm = _compute_uiconm(image)
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm


def compute_uiqm_parts(image):
    uicm = _compute_uicm(image)
    uism = _compute_uism(image)
    uiconm = _compute_uiconm(image)
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return float(uiqm), float(uicm), float(uism), float(uiconm)


def _ssim_single_channel(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    # 高斯核平滑
    kernel_size = (11, 11)
    sigma = 1.5
    ux = cv2.GaussianBlur(x, kernel_size, sigma)
    uy = cv2.GaussianBlur(y, kernel_size, sigma)
    uxx = cv2.GaussianBlur(x * x, kernel_size, sigma)
    uyy = cv2.GaussianBlur(y * y, kernel_size, sigma)
    uxy = cv2.GaussianBlur(x * y, kernel_size, sigma)
    sx2 = uxx - ux * ux
    sy2 = uyy - uy * uy
    sxy = uxy - ux * uy
    num = (2 * ux * uy + C1) * (2 * sxy + C2)
    den = (ux ** 2 + uy ** 2 + C1) * (sx2 + sy2 + C2)
    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean())


def compute_ssim(img1, img2):
    # 兼容无 cv2.quality 的环境；对彩色图逐通道计算并取平均
    if img1.ndim == 3 and img1.shape[2] == 3:
        s = 0.0
        for c in range(3):
            s += _ssim_single_channel(img1[:, :, c], img2[:, :, c])
        return s / 3.0
    else:
        return _ssim_single_channel(img1, img2)


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _is_image_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _normalize_key(stem: str) -> str:
    s = stem
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
        "__gt",
        "_gt",
        "-gt",
        " gt",
        "__gtr",
        "_gtr",
        "-gtr",
        " gtr",
        "__ref",
        "_ref",
        "-ref",
        " ref",
        "__reference",
        "_reference",
        "-reference",
        " reference",
        "__target",
        "_target",
        "-target",
        " target",
    ]:
        idx = lower.find(token)
        if idx != -1:
            s = s[:idx]
            lower = s.lower()
    s = s.strip()
    lower = s.lower()
    for prefix in ["raw_", "input_", "inp_", "underwater_", "uw_", "ref_", "gt_", "gtr_"]:
        if lower.startswith(prefix):
            s = s[len(prefix):].strip()
            lower = s.lower()
    s = s.strip()
    return s.rstrip("_- ")


def _index_images_by_key(dir_path: Path):
    by_key = {}
    for p in sorted([x for x in dir_path.glob("**/*") if _is_image_file(x)]):
        key = _normalize_key(p.stem)
        by_key.setdefault(key, []).append(p)
    return by_key


def _pick_best_pred(candidates):
    if not candidates:
        return None
    scored = []
    for p in candidates:
        name = p.name.lower()
        score = 0
        if "fake" in name:
            score += 10
        if "real" in name:
            score -= 10
        if p.suffix.lower() == ".png":
            score += 2
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1].name))
    return scored[0][1]


def _bootstrap_ci(values, iters: int, seed: int):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    if iters <= 0:
        mean = float(arr.mean())
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975]).tolist()
    return (float(arr.mean()), float(lo), float(hi))


def _benchmark_fieldnames():
    return [
        "dataset",
        "model",
        "epoch",
        "matched",
        "psnr",
        "psnr_ci_low",
        "psnr_ci_high",
        "ssim",
        "ssim_ci_low",
        "ssim_ci_high",
        "uciqe_inp",
        "uciqe_pred",
        "uciqe_ref",
        "uiqm_inp",
        "uiqm_pred",
        "uiqm_ref",
        "uicm_inp",
        "uicm_pred",
        "uicm_ref",
        "uism_inp",
        "uism_pred",
        "uism_ref",
        "uiconm_inp",
        "uiconm_pred",
        "uiconm_ref",
    ]


def _summary_to_benchmark_row(summary: dict, dataset: str, model: str, epoch: str) -> dict:
    row = {
        "dataset": dataset,
        "model": model,
        "epoch": epoch,
        "matched": summary.get("matched", ""),
        "psnr": "",
        "psnr_ci_low": "",
        "psnr_ci_high": "",
        "ssim": "",
        "ssim_ci_low": "",
        "ssim_ci_high": "",
        "uciqe_inp": summary.get("uciqe_inp", ""),
        "uciqe_pred": summary.get("uciqe_pred", ""),
        "uciqe_ref": summary.get("uciqe_ref", ""),
        "uiqm_inp": summary.get("uiqm_inp", ""),
        "uiqm_pred": summary.get("uiqm_pred", ""),
        "uiqm_ref": summary.get("uiqm_ref", ""),
        "uicm_inp": summary.get("uicm_inp", ""),
        "uicm_pred": summary.get("uicm_pred", ""),
        "uicm_ref": summary.get("uicm_ref", ""),
        "uism_inp": summary.get("uism_inp", ""),
        "uism_pred": summary.get("uism_pred", ""),
        "uism_ref": summary.get("uism_ref", ""),
        "uiconm_inp": summary.get("uiconm_inp", ""),
        "uiconm_pred": summary.get("uiconm_pred", ""),
        "uiconm_ref": summary.get("uiconm_ref", ""),
    }
    if "psnr" in summary and isinstance(summary["psnr"], dict):
        row["psnr"] = summary["psnr"].get("mean", "")
        row["psnr_ci_low"] = summary["psnr"].get("ci95_low", "")
        row["psnr_ci_high"] = summary["psnr"].get("ci95_high", "")
    if "ssim" in summary and isinstance(summary["ssim"], dict):
        row["ssim"] = summary["ssim"].get("mean", "")
        row["ssim_ci_low"] = summary["ssim"].get("ci95_low", "")
        row["ssim_ci_high"] = summary["ssim"].get("ci95_high", "")
    return row


def _parse_epoch_key(epoch: str):
    if epoch is None:
        return (1, 0)
    s = str(epoch).strip()
    if s == "":
        return (1, 0)
    if s.lower() == "latest":
        return (1, 10**9)
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def _upsert_benchmark_csv(csv_path: Path, row: dict):
    fieldnames = _benchmark_fieldnames()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing.append(r)

    def key(r):
        return (str(r.get("dataset", "")), str(r.get("model", "")), str(r.get("epoch", "")))

    target_key = key(row)
    replaced = False
    merged = []
    for r in existing:
        if key(r) == target_key:
            merged.append({k: row.get(k, "") for k in fieldnames})
            replaced = True
        else:
            merged.append({k: r.get(k, "") for k in fieldnames})
    if not replaced:
        merged.append({k: row.get(k, "") for k in fieldnames})

    merged.sort(key=lambda r: (r.get("dataset", ""), r.get("model", ""), _parse_epoch_key(r.get("epoch", ""))))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_dir", required=True)
    parser.add_argument("--gtr_dir", required=False)
    parser.add_argument("--ref_dir", required=False)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--save_csv", required=False)
    parser.add_argument("--save_json", required=False)
    parser.add_argument("--benchmark_csv", required=False)
    parser.add_argument("--benchmark_dataset", required=False, default="")
    parser.add_argument("--benchmark_model", required=False, default="")
    parser.add_argument("--benchmark_epoch", required=False, default="")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--report_uiqm_parts", action="store_true")
    args = parser.parse_args()

    inp_dir = Path(args.inp_dir)
    pred_dir = Path(args.pred_dir)
    ref_dir_str = args.ref_dir if args.ref_dir else args.gtr_dir
    has_ref = ref_dir_str is not None and ref_dir_str != ""
    ref_dir = Path(ref_dir_str) if has_ref else None

    inp_files = sorted([p for p in inp_dir.glob("**/*") if _is_image_file(p)])
    if args.max_images and args.max_images > 0:
        inp_files = inp_files[: int(args.max_images)]
    pred_by_key = _index_images_by_key(pred_dir)
    ref_by_key = _index_images_by_key(ref_dir) if has_ref else {}

    psnrs = []
    ssims = []
    uciqe_inp = []
    uiqm_inp = []
    uciqe_pred = []
    uiqm_pred = []
    uciqe_ref = []
    uiqm_ref = []
    uicm_inp = []
    uicm_pred = []
    uicm_ref = []
    uism_inp = []
    uism_pred = []
    uism_ref = []
    uiconm_inp = []
    uiconm_pred = []
    uiconm_ref = []
    matched = 0
    rows = []

    for inp_path in inp_files:
        key = _normalize_key(inp_path.stem)
        pred_candidates = pred_by_key.get(key, [])
        pred_path = _pick_best_pred(pred_candidates)
        if pred_path is None:
            candidates = list(pred_dir.glob(f"{inp_path.stem}*_fake*.png"))
            if len(candidates) == 0:
                candidates = list(pred_dir.glob(f"{inp_path.stem}*_fake*.jpg"))
            pred_path = _pick_best_pred(candidates)
        if pred_path is None:
            continue

        inp_img = read_image(inp_path)
        pred_img = read_image(pred_path)
        ref_img = None
        ref_path = None
        if has_ref:
            ref_candidates = ref_by_key.get(key, [])
            if not ref_candidates:
                ref_candidates = list(ref_dir.glob(f"{inp_path.stem}*"))
            if ref_candidates:
                ref_path = sorted(ref_candidates)[0]
                ref_img = read_image(ref_path)

        h, w = inp_img.shape[:2]
        if pred_img.shape[:2] != (h, w):
            pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_AREA)
        if ref_img is not None and ref_img.shape[:2] != (h, w):
            ref_img = cv2.resize(ref_img, (w, h), interpolation=cv2.INTER_AREA)

        psnr_val = None
        ssim_val = None
        if ref_img is not None:
            psnr_val = compute_psnr(ref_img, pred_img)
            ssim_val = compute_ssim(ref_img, pred_img)
            psnrs.append(psnr_val)
            ssims.append(ssim_val)
        uciqe_inp_val = compute_uciqe(inp_img)
        uciqe_pred_val = compute_uciqe(pred_img)
        if args.report_uiqm_parts:
            uiqm_inp_val, uicm_inp_val, uism_inp_val, uiconm_inp_val = compute_uiqm_parts(inp_img)
            uiqm_pred_val, uicm_pred_val, uism_pred_val, uiconm_pred_val = compute_uiqm_parts(pred_img)
            uicm_inp.append(uicm_inp_val)
            uicm_pred.append(uicm_pred_val)
            uism_inp.append(uism_inp_val)
            uism_pred.append(uism_pred_val)
            uiconm_inp.append(uiconm_inp_val)
            uiconm_pred.append(uiconm_pred_val)
        else:
            uiqm_inp_val = compute_uiqm(inp_img)
            uiqm_pred_val = compute_uiqm(pred_img)
        uciqe_inp.append(uciqe_inp_val)
        uiqm_inp.append(uiqm_inp_val)
        uciqe_pred.append(uciqe_pred_val)
        uiqm_pred.append(uiqm_pred_val)
        uciqe_ref_val = None
        uiqm_ref_val = None
        if ref_img is not None:
            uciqe_ref_val = compute_uciqe(ref_img)
            if args.report_uiqm_parts:
                uiqm_ref_val, uicm_ref_val, uism_ref_val, uiconm_ref_val = compute_uiqm_parts(ref_img)
                uicm_ref.append(uicm_ref_val)
                uism_ref.append(uism_ref_val)
                uiconm_ref.append(uiconm_ref_val)
            else:
                uiqm_ref_val = compute_uiqm(ref_img)
            uciqe_ref.append(uciqe_ref_val)
            uiqm_ref.append(uiqm_ref_val)
        matched += 1
        row = {
            "key": key,
            "inp_path": str(inp_path),
            "pred_path": str(pred_path),
            "ref_path": str(ref_path) if ref_path is not None else "",
            "psnr": float(psnr_val) if psnr_val is not None else "",
            "ssim": float(ssim_val) if ssim_val is not None else "",
            "uciqe_inp": float(uciqe_inp_val),
            "uciqe_pred": float(uciqe_pred_val),
            "uciqe_ref": float(uciqe_ref_val) if uciqe_ref_val is not None else "",
            "uiqm_inp": float(uiqm_inp_val),
            "uiqm_pred": float(uiqm_pred_val),
            "uiqm_ref": float(uiqm_ref_val) if uiqm_ref_val is not None else "",
        }
        if args.report_uiqm_parts:
            row.update(
                {
                    "uicm_inp": float(uicm_inp_val),
                    "uicm_pred": float(uicm_pred_val),
                    "uicm_ref": float(uicm_ref_val) if ref_img is not None else "",
                    "uism_inp": float(uism_inp_val),
                    "uism_pred": float(uism_pred_val),
                    "uism_ref": float(uism_ref_val) if ref_img is not None else "",
                    "uiconm_inp": float(uiconm_inp_val),
                    "uiconm_pred": float(uiconm_pred_val),
                    "uiconm_ref": float(uiconm_ref_val) if ref_img is not None else "",
                }
            )
        rows.append(row)

    if matched == 0:
        print("No matched predictions found by name; falling back to index-based pairing.")
        pred_files = sorted([p for p in pred_dir.glob("**/*") if _is_image_file(p)])
        n = min(len(inp_files), len(pred_files))
        if n == 0:
            print("No predictions available for index-based pairing.")
            return
        psnrs = []
        ssims = []
        uciqe_inp = []
        uiqm_inp = []
        uciqe_pred = []
        uiqm_pred = []
        uciqe_ref = []
        uiqm_ref = []
        uicm_inp = []
        uicm_pred = []
        uicm_ref = []
        uism_inp = []
        uism_pred = []
        uism_ref = []
        uiconm_inp = []
        uiconm_pred = []
        uiconm_ref = []
        matched = 0
        rows = []
        for i in range(n):
            inp_path = inp_files[i]
            pred_path = pred_files[i]
            key = _normalize_key(inp_path.stem)
            inp_img = read_image(inp_path)
            pred_img = read_image(pred_path)
            ref_img = None
            ref_path = None
            if has_ref:
                ref_candidates = ref_by_key.get(key, [])
                if not ref_candidates:
                    ref_candidates = list(ref_dir.glob(f"{inp_path.stem}*"))
                if ref_candidates:
                    ref_path = sorted(ref_candidates)[0]
                    ref_img = read_image(ref_path)
            h, w = inp_img.shape[:2]
            if pred_img.shape[:2] != (h, w):
                pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_AREA)
            if ref_img is not None and ref_img.shape[:2] != (h, w):
                ref_img = cv2.resize(ref_img, (w, h), interpolation=cv2.INTER_AREA)
            psnr_val = None
            ssim_val = None
            if ref_img is not None:
                psnr_val = compute_psnr(ref_img, pred_img)
                ssim_val = compute_ssim(ref_img, pred_img)
                psnrs.append(psnr_val)
                ssims.append(ssim_val)
            uciqe_inp_val = compute_uciqe(inp_img)
            uciqe_pred_val = compute_uciqe(pred_img)
            if args.report_uiqm_parts:
                uiqm_inp_val, uicm_inp_val, uism_inp_val, uiconm_inp_val = compute_uiqm_parts(inp_img)
                uiqm_pred_val, uicm_pred_val, uism_pred_val, uiconm_pred_val = compute_uiqm_parts(pred_img)
                uicm_inp.append(uicm_inp_val)
                uicm_pred.append(uicm_pred_val)
                uism_inp.append(uism_inp_val)
                uism_pred.append(uism_pred_val)
                uiconm_inp.append(uiconm_inp_val)
                uiconm_pred.append(uiconm_pred_val)
            else:
                uiqm_inp_val = compute_uiqm(inp_img)
                uiqm_pred_val = compute_uiqm(pred_img)
            uciqe_inp.append(uciqe_inp_val)
            uiqm_inp.append(uiqm_inp_val)
            uciqe_pred.append(uciqe_pred_val)
            uiqm_pred.append(uiqm_pred_val)
            uciqe_ref_val = None
            uiqm_ref_val = None
            if ref_img is not None:
                uciqe_ref_val = compute_uciqe(ref_img)
                if args.report_uiqm_parts:
                    uiqm_ref_val, uicm_ref_val, uism_ref_val, uiconm_ref_val = compute_uiqm_parts(ref_img)
                    uicm_ref.append(uicm_ref_val)
                    uism_ref.append(uism_ref_val)
                    uiconm_ref.append(uiconm_ref_val)
                else:
                    uiqm_ref_val = compute_uiqm(ref_img)
                uciqe_ref.append(uciqe_ref_val)
                uiqm_ref.append(uiqm_ref_val)
            matched += 1
            row = {
                "key": key,
                "inp_path": str(inp_path),
                "pred_path": str(pred_path),
                "ref_path": str(ref_path) if ref_path is not None else "",
                "psnr": float(psnr_val) if psnr_val is not None else "",
                "ssim": float(ssim_val) if ssim_val is not None else "",
                "uciqe_inp": float(uciqe_inp_val),
                "uciqe_pred": float(uciqe_pred_val),
                "uciqe_ref": float(uciqe_ref_val) if uciqe_ref_val is not None else "",
                "uiqm_inp": float(uiqm_inp_val),
                "uiqm_pred": float(uiqm_pred_val),
                "uiqm_ref": float(uiqm_ref_val) if uiqm_ref_val is not None else "",
            }
            if args.report_uiqm_parts:
                row.update(
                    {
                        "uicm_inp": float(uicm_inp_val),
                        "uicm_pred": float(uicm_pred_val),
                        "uicm_ref": float(uicm_ref_val) if ref_img is not None else "",
                        "uism_inp": float(uism_inp_val),
                        "uism_pred": float(uism_pred_val),
                        "uism_ref": float(uism_ref_val) if ref_img is not None else "",
                        "uiconm_inp": float(uiconm_inp_val),
                        "uiconm_pred": float(uiconm_pred_val),
                        "uiconm_ref": float(uiconm_ref_val) if ref_img is not None else "",
                    }
                )
            rows.append(row)

    print(f"Matched pairs: {matched}")
    summary = {"matched": matched}
    if has_ref and len(psnrs) > 0:
        psnr_ci = _bootstrap_ci(psnrs, args.bootstrap_iters, args.seed)
        ssim_ci = _bootstrap_ci(ssims, args.bootstrap_iters, args.seed + 1)
        summary["psnr"] = {"mean": psnr_ci[0], "ci95_low": psnr_ci[1], "ci95_high": psnr_ci[2]}
        summary["ssim"] = {"mean": ssim_ci[0], "ci95_low": ssim_ci[1], "ci95_high": ssim_ci[2]}
        print(f"Average PSNR: {psnr_ci[0]:.4f} dB")
        print(f"PSNR 95% CI: [{psnr_ci[1]:.4f}, {psnr_ci[2]:.4f}]")
        print(f"Average SSIM: {ssim_ci[0]:.4f}")
        print(f"SSIM 95% CI: [{ssim_ci[1]:.4f}, {ssim_ci[2]:.4f}]")
    summary["uciqe_inp"] = float(np.mean(uciqe_inp))
    summary["uciqe_pred"] = float(np.mean(uciqe_pred))
    summary["uiqm_inp"] = float(np.mean(uiqm_inp))
    summary["uiqm_pred"] = float(np.mean(uiqm_pred))
    print(f"Average UCIQE (Inp): {summary['uciqe_inp']:.4f}")
    print(f"Average UCIQE (Pred): {summary['uciqe_pred']:.4f}")
    if has_ref and len(uciqe_ref) > 0:
        summary["uciqe_ref"] = float(np.mean(uciqe_ref))
        print(f"Average UCIQE (GTr): {summary['uciqe_ref']:.4f}")
    print(f"Average UIQM (Inp): {summary['uiqm_inp']:.4f}")
    print(f"Average UIQM (Pred): {summary['uiqm_pred']:.4f}")
    if has_ref and len(uiqm_ref) > 0:
        summary["uiqm_ref"] = float(np.mean(uiqm_ref))
        print(f"Average UIQM (GTr): {summary['uiqm_ref']:.4f}")
    if args.report_uiqm_parts and matched > 0:
        summary["uicm_inp"] = float(np.mean(uicm_inp)) if uicm_inp else None
        summary["uicm_pred"] = float(np.mean(uicm_pred)) if uicm_pred else None
        summary["uism_inp"] = float(np.mean(uism_inp)) if uism_inp else None
        summary["uism_pred"] = float(np.mean(uism_pred)) if uism_pred else None
        summary["uiconm_inp"] = float(np.mean(uiconm_inp)) if uiconm_inp else None
        summary["uiconm_pred"] = float(np.mean(uiconm_pred)) if uiconm_pred else None
        print(f"Average UICM (Inp): {summary['uicm_inp']:.4f}")
        print(f"Average UICM (Pred): {summary['uicm_pred']:.4f}")
        print(f"Average UISM (Inp): {summary['uism_inp']:.4f}")
        print(f"Average UISM (Pred): {summary['uism_pred']:.4f}")
        print(f"Average UIConM (Inp): {summary['uiconm_inp']:.4f}")
        print(f"Average UIConM (Pred): {summary['uiconm_pred']:.4f}")
        if has_ref and len(uiqm_ref) > 0:
            summary["uicm_ref"] = float(np.mean(uicm_ref)) if uicm_ref else None
            summary["uism_ref"] = float(np.mean(uism_ref)) if uism_ref else None
            summary["uiconm_ref"] = float(np.mean(uiconm_ref)) if uiconm_ref else None
            print(f"Average UICM (GTr): {summary['uicm_ref']:.4f}")
            print(f"Average UISM (GTr): {summary['uism_ref']:.4f}")
            print(f"Average UIConM (GTr): {summary['uiconm_ref']:.4f}")

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "key",
            "inp_path",
            "pred_path",
            "ref_path",
            "psnr",
            "ssim",
            "uciqe_inp",
            "uciqe_pred",
            "uciqe_ref",
            "uiqm_inp",
            "uiqm_pred",
            "uiqm_ref",
        ]
        if args.report_uiqm_parts:
            fieldnames += [
                "uicm_inp",
                "uicm_pred",
                "uicm_ref",
                "uism_inp",
                "uism_pred",
                "uism_ref",
                "uiconm_inp",
                "uiconm_pred",
                "uiconm_ref",
            ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("Saved per-image metrics to", out_path)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Saved summary to", out_path)

    if args.benchmark_csv:
        if not args.benchmark_dataset or not args.benchmark_model:
            raise SystemExit("--benchmark_dataset and --benchmark_model are required when using --benchmark_csv")
        csv_path = Path(args.benchmark_csv)
        row = _summary_to_benchmark_row(
            summary=summary,
            dataset=str(args.benchmark_dataset),
            model=str(args.benchmark_model),
            epoch=str(args.benchmark_epoch),
        )
        _upsert_benchmark_csv(csv_path, row)
        print("Updated benchmark CSV at", csv_path)


if __name__ == "__main__":
    main()
