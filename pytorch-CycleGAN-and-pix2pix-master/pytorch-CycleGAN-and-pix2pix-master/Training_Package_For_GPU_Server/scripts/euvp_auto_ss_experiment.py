import subprocess
import sys
import re
import csv
import json
from pathlib import Path
import argparse
import os
import shutil
import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
DATA_ROOT = Path(r"d:\VScode\Graduation project\EUVP_Unpaired")
TEST_INP = Path(r"d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp")
TEST_GTR = Path(r"d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr")


def run(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_capture(cmd):
    print("Running (capture):", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout


def train_ss_tuned():
    name = "euvp_cyclegan_ss_tuned50"
    cmd = [
        sys.executable,
        "train.py",
        "--dataroot",
        str(DATA_ROOT),
        "--name",
        name,
        "--model",
        "cycle_gan",
        "--n_epochs",
        "50",
        "--n_epochs_decay",
        "0",
        "--lambda_color",
        "0.2",
        "--lambda_struct",
        "2.0",
        "--lambda_perceptual",
        "0.05",
        "--perceptual_layer",
        "16",
        "--perceptual_weights",
        "imagenet",
    ]
    run(cmd)
    return name


def test_model(name, epoch):
    cmd = [
        sys.executable,
        "test.py",
        "--dataroot",
        str(TEST_INP),
        "--name",
        name,
        "--model",
        "test",
        "--dataset_mode",
        "single",
        "--model_suffix",
        "_A",
        "--epoch",
        str(epoch),
        "--num_test",
        "200",
    ]
    run(cmd)


def eval_pred_dir(model_name, pred_dir):
    cmd = [
        sys.executable,
        "scripts/evaluate_euvp_psnr_ssim.py",
        "--inp_dir",
        str(TEST_INP),
        "--gtr_dir",
        str(TEST_GTR),
        "--pred_dir",
        str(pred_dir),
    ]
    out = run_capture(cmd)
    psnr = None
    ssim = None
    uciqe_pred = None
    uiqm_pred = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Average PSNR:"):
            m = re.search(r"Average PSNR:\s*([0-9.]+)", line)
            if m:
                psnr = float(m.group(1))
        elif line.startswith("Average SSIM:"):
            m = re.search(r"Average SSIM:\s*([0-9.]+)", line)
            if m:
                ssim = float(m.group(1))
        elif line.startswith("Average UCIQE (Pred):"):
            m = re.search(r"Average UCIQE \(Pred\):\s*([0-9.]+)", line)
            if m:
                uciqe_pred = float(m.group(1))
        elif line.startswith("Average UIQM (Pred):"):
            m = re.search(r"Average UIQM \(Pred\):\s*([0-9.]+)", line)
            if m:
                uiqm_pred = float(m.group(1))
    return {
        "model": model_name,
        "psnr": psnr,
        "ssim": ssim,
        "uciqe_pred": uciqe_pred,
        "uiqm_pred": uiqm_pred,
    }


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _write_png(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise ValueError(f"Failed to write image: {path}")


def _gray_world_wb(img):
    x = img.astype(np.float32)
    mean = x.reshape(-1, 3).mean(axis=0)
    mean_gray = float(mean.mean())
    scale = mean_gray / (mean + 1e-6)
    y = x * scale.reshape(1, 1, 3)
    return np.clip(y, 0, 255).astype(np.uint8)


def _clahe_lab(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _gamma(img, gamma=1.2):
    g = float(gamma)
    inv = 1.0 / g
    table = (np.linspace(0, 1, 256) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)


def _first_existing_dir(paths: list[Path]) -> Path | None:
    for p in paths:
        if p is not None and p.exists() and p.is_dir():
            return p
    return None


def _infer_uieb_dirs(root: Path) -> tuple[Path | None, Path | None]:
    inp_names = ["raw", "input", "inp", "underwater", "underwater_images"]
    ref_names = ["reference", "ref", "gt", "gtr", "target", "enhanced", "reference_images"]

    direct_inp = _first_existing_dir([root / n for n in inp_names])
    direct_ref = _first_existing_dir([root / n for n in ref_names])
    if direct_inp is not None or direct_ref is not None:
        return direct_inp, direct_ref

    try:
        subdirs = [p for p in root.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return None, None

    for d in subdirs:
        inp = _first_existing_dir([d / n for n in inp_names])
        ref = _first_existing_dir([d / n for n in ref_names])
        if inp is not None:
            return inp, ref
    return None, None


def _find_uieb_root() -> Path | None:
    candidates = [
        WORKSPACE_ROOT / "UIEB Dataset",
        WORKSPACE_ROOT / "UIEB_Dataset",
        WORKSPACE_ROOT / "UIEB",
        WORKSPACE_ROOT / "datasets" / "UIEB",
        WORKSPACE_ROOT / "dataset" / "UIEB",
        WORKSPACE_ROOT / "data" / "UIEB",
    ]
    return _first_existing_dir(candidates)


def generate_baseline_outputs(inp_dir: Path, out_dir: Path, method: str, max_images: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    inp_files = sorted([p for p in inp_dir.glob("**/*") if _is_image_file(p)])
    if max_images and max_images > 0:
        inp_files = inp_files[: int(max_images)]
    for p in inp_files:
        img = _read_bgr(p)
        if method == "identity":
            out = img
        elif method == "grayworld":
            out = _gray_world_wb(img)
        elif method == "clahe":
            out = _clahe_lab(img)
        elif method == "gamma":
            out = _gamma(img)
        elif method == "grayworld_clahe":
            out = _clahe_lab(_gray_world_wb(img))
        else:
            raise ValueError(f"Unknown baseline method: {method}")
        _write_png(out_dir / f"{p.stem}__fake.png", out)


def run_test_single(name: str, epoch: str, inp_dir: Path, num_test: int, model_suffix: str, gpu_ids: str):
    ckpt_dir = REPO_ROOT / "checkpoints" / name
    ckpt_path = ckpt_dir / f"{epoch}_net_G{model_suffix}.pth"
    if not ckpt_path.exists():
        alt = ckpt_dir / f"{epoch}_net_G.pth"
        if alt.exists():
            ckpt_path = alt
        else:
            print(f"Skip inference: checkpoint not found for {name} epoch={epoch} (expected {ckpt_path})")
            return False

    env = os.environ.copy()
    if str(gpu_ids).strip() == "-1":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip()
    cmd = [
        sys.executable,
        "test.py",
        "--dataroot",
        str(inp_dir),
        "--name",
        name,
        "--model",
        "test",
        "--dataset_mode",
        "single",
        "--model_suffix",
        model_suffix,
        "--epoch",
        str(epoch),
        "--num_test",
        str(num_test),
    ]
    print("Running:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return True


def parse_eval_output(text: str):
    out = {"matched": None, "psnr": None, "psnr_ci_low": None, "psnr_ci_high": None, "ssim": None, "ssim_ci_low": None, "ssim_ci_high": None}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Matched pairs:"):
            m = re.search(r"Matched pairs:\s*(\d+)", s)
            if m:
                out["matched"] = int(m.group(1))
        elif s.startswith("Average PSNR:"):
            m = re.search(r"Average PSNR:\s*([0-9.]+)", s)
            if m:
                out["psnr"] = float(m.group(1))
        elif s.startswith("PSNR 95% CI:"):
            m = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", s)
            if m:
                out["psnr_ci_low"] = float(m.group(1))
                out["psnr_ci_high"] = float(m.group(2))
        elif s.startswith("Average SSIM:"):
            m = re.search(r"Average SSIM:\s*([0-9.]+)", s)
            if m:
                out["ssim"] = float(m.group(1))
        elif s.startswith("SSIM 95% CI:"):
            m = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", s)
            if m:
                out["ssim_ci_low"] = float(m.group(1))
                out["ssim_ci_high"] = float(m.group(2))
        elif s.startswith("Average UCIQE (Inp):"):
            m = re.search(r"Average UCIQE \(Inp\):\s*([0-9.]+)", s)
            if m:
                out["uciqe_inp"] = float(m.group(1))
        elif s.startswith("Average UCIQE (Pred):"):
            m = re.search(r"Average UCIQE \(Pred\):\s*([0-9.]+)", s)
            if m:
                out["uciqe_pred"] = float(m.group(1))
        elif s.startswith("Average UCIQE (GTr):"):
            m = re.search(r"Average UCIQE \(GTr\):\s*([0-9.]+)", s)
            if m:
                out["uciqe_ref"] = float(m.group(1))
        elif s.startswith("Average UIQM (Inp):"):
            m = re.search(r"Average UIQM \(Inp\):\s*([0-9.]+)", s)
            if m:
                out["uiqm_inp"] = float(m.group(1))
        elif s.startswith("Average UIQM (Pred):"):
            m = re.search(r"Average UIQM \(Pred\):\s*([0-9.]+)", s)
            if m:
                out["uiqm_pred"] = float(m.group(1))
        elif s.startswith("Average UIQM (GTr):"):
            m = re.search(r"Average UIQM \(GTr\):\s*([0-9.]+)", s)
            if m:
                out["uiqm_ref"] = float(m.group(1))
        elif s.startswith("Average UICM (Inp):"):
            m = re.search(r"Average UICM \(Inp\):\s*([0-9.]+)", s)
            if m:
                out["uicm_inp"] = float(m.group(1))
        elif s.startswith("Average UICM (Pred):"):
            m = re.search(r"Average UICM \(Pred\):\s*([0-9.]+)", s)
            if m:
                out["uicm_pred"] = float(m.group(1))
        elif s.startswith("Average UICM (GTr):"):
            m = re.search(r"Average UICM \(GTr\):\s*([0-9.]+)", s)
            if m:
                out["uicm_ref"] = float(m.group(1))
        elif s.startswith("Average UISM (Inp):"):
            m = re.search(r"Average UISM \(Inp\):\s*([0-9.]+)", s)
            if m:
                out["uism_inp"] = float(m.group(1))
        elif s.startswith("Average UISM (Pred):"):
            m = re.search(r"Average UISM \(Pred\):\s*([0-9.]+)", s)
            if m:
                out["uism_pred"] = float(m.group(1))
        elif s.startswith("Average UISM (GTr):"):
            m = re.search(r"Average UISM \(GTr\):\s*([0-9.]+)", s)
            if m:
                out["uism_ref"] = float(m.group(1))
        elif s.startswith("Average UIConM (Inp):"):
            m = re.search(r"Average UIConM \(Inp\):\s*([0-9.]+)", s)
            if m:
                out["uiconm_inp"] = float(m.group(1))
        elif s.startswith("Average UIConM (Pred):"):
            m = re.search(r"Average UIConM \(Pred\):\s*([0-9.]+)", s)
            if m:
                out["uiconm_pred"] = float(m.group(1))
        elif s.startswith("Average UIConM (GTr):"):
            m = re.search(r"Average UIConM \(GTr\):\s*([0-9.]+)", s)
            if m:
                out["uiconm_ref"] = float(m.group(1))
    return out


def run_evaluate(inp_dir: Path, ref_dir: Path | None, pred_dir: Path, save_csv: Path | None, save_json: Path | None, bootstrap_iters: int, seed: int, max_images: int, report_uiqm_parts: bool):
    cmd = [
        sys.executable,
        "scripts/evaluate_euvp_psnr_ssim.py",
        "--inp_dir",
        str(inp_dir),
        "--pred_dir",
        str(pred_dir),
        "--bootstrap_iters",
        str(bootstrap_iters),
        "--seed",
        str(seed),
    ]
    if max_images and max_images > 0:
        cmd += ["--max_images", str(max_images)]
    if ref_dir is not None:
        cmd += ["--ref_dir", str(ref_dir)]
    if report_uiqm_parts:
        cmd += ["--report_uiqm_parts"]
    if save_csv is not None:
        cmd += ["--save_csv", str(save_csv)]
    if save_json is not None:
        cmd += ["--save_json", str(save_json)]
    out = run_capture(cmd)
    return out, parse_eval_output(out)


def _metrics_from_summary_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {"matched": data.get("matched")}
    psnr = data.get("psnr")
    if isinstance(psnr, dict):
        out["psnr"] = psnr.get("mean")
        out["psnr_ci_low"] = psnr.get("ci95_low")
        out["psnr_ci_high"] = psnr.get("ci95_high")
    ssim = data.get("ssim")
    if isinstance(ssim, dict):
        out["ssim"] = ssim.get("mean")
        out["ssim_ci_low"] = ssim.get("ci95_low")
        out["ssim_ci_high"] = ssim.get("ci95_high")
    for k in ["uciqe_inp", "uciqe_pred", "uciqe_ref", "uiqm_inp", "uiqm_pred", "uiqm_ref"]:
        if k in data:
            out[k] = data.get(k)
    return out


def _iter_existing_benchmark_summaries(bench_dir: Path) -> list[dict]:
    rows = []
    for p in bench_dir.rglob("*summary.json"):
        try:
            parts = list(p.parts)
            idx = parts.index("benchmarks")
        except ValueError:
            continue
        dataset = parts[idx + 1] if len(parts) > idx + 1 else ""
        rel = parts[idx + 2:]
        if not rel:
            continue

        if p.name == "summary.json":
            if len(rel) >= 4 and rel[0] == "models":
                model = rel[1]
                epoch = rel[2].replace("epoch_", "")
                metrics = _metrics_from_summary_json(p)
                metrics.update({"dataset": dataset, "model": model, "epoch": epoch})
                rows.append(metrics)
        else:
            if len(rel) >= 2 and rel[0] == "baselines" and p.name.endswith("_summary.json"):
                method = p.stem[:-len("_summary")]
                metrics = _metrics_from_summary_json(p)
                metrics.update({"dataset": dataset, "model": f"baseline:{method}", "epoch": ""})
                rows.append(metrics)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="euvp_cyclegan_full")
    parser.add_argument("--epochs", default="50,100,150,200")
    parser.add_argument("--datasets", default="euvp")
    parser.add_argument("--euvp_inp", default=str(TEST_INP))
    parser.add_argument("--euvp_ref", default=str(TEST_GTR))
    parser.add_argument("--uieb_root", default="")
    parser.add_argument("--uieb_inp", default="")
    parser.add_argument("--uieb_ref", default="")
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--model_suffix", default="_A")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--include_baselines", action="store_true")
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--report_uiqm_parts", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    epochs = [e.strip() for e in args.epochs.split(",") if e.strip()]
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    dataset_cfg = {}
    if "euvp" in datasets:
        dataset_cfg["euvp"] = {"inp": Path(args.euvp_inp), "ref": Path(args.euvp_ref) if args.euvp_ref else None}
    if "uieb" in datasets:
        if args.uieb_inp:
            dataset_cfg["uieb"] = {"inp": Path(args.uieb_inp), "ref": Path(args.uieb_ref) if args.uieb_ref else None}
        else:
            uieb_root = Path(args.uieb_root) if args.uieb_root else _find_uieb_root()
            if uieb_root is not None:
                inp_dir, ref_dir = _infer_uieb_dirs(uieb_root)
                dataset_cfg["uieb"] = {"inp": inp_dir, "ref": ref_dir}
            else:
                dataset_cfg["uieb"] = {"inp": None, "ref": None}

    bench_dir = REPO_ROOT / "results" / "benchmarks"
    if bench_dir.exists():
        bench_dir.mkdir(parents=True, exist_ok=True)
    else:
        bench_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for dname, cfg in dataset_cfg.items():
        inp_dir = cfg.get("inp")
        ref_dir = cfg.get("ref")
        if inp_dir is None or not inp_dir.exists():
            print(f"Skip dataset {dname}: input dir not found")
            continue
        if ref_dir is not None and not ref_dir.exists():
            print(f"Skip dataset {dname}: ref dir not found")
            continue

        if args.include_baselines:
            for method in ["identity", "grayworld", "clahe", "gamma", "grayworld_clahe"]:
                out_dir = bench_dir / dname / "baselines" / method / "images"
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                generate_baseline_outputs(inp_dir, out_dir, method, args.num_test)
                per_csv = bench_dir / dname / "baselines" / f"{method}_per_image.csv"
                per_json = bench_dir / dname / "baselines" / f"{method}_summary.json"
                _, metrics = run_evaluate(inp_dir, ref_dir, out_dir, per_csv, per_json, args.bootstrap_iters, args.seed, args.num_test, args.report_uiqm_parts)
                metrics.update({"dataset": dname, "model": f"baseline:{method}", "epoch": ""})
                summary_rows.append(metrics)

        for model in models:
            for epoch in epochs:
                pred_dir = REPO_ROOT / "results" / model / f"test_{epoch}" / "images"
                if args.run_inference or not pred_dir.exists():
                    ok = run_test_single(model, epoch, inp_dir, args.num_test, args.model_suffix, args.gpu_ids)
                    if not ok:
                        continue
                per_dir = bench_dir / dname / "models" / model / f"epoch_{epoch}"
                per_csv = per_dir / "per_image.csv"
                per_json = per_dir / "summary.json"
                _, metrics = run_evaluate(inp_dir, ref_dir, pred_dir, per_csv, per_json, args.bootstrap_iters, args.seed, args.num_test, args.report_uiqm_parts)
                metrics.update({"dataset": dname, "model": model, "epoch": epoch})
                summary_rows.append(metrics)

    out_csv = bench_dir / "summary.csv"
    if not summary_rows:
        summary_rows = _iter_existing_benchmark_summaries(bench_dir)
    if not summary_rows:
        print("No benchmark rows generated; keep existing summary.csv")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print("Saved summary to", out_csv)


if __name__ == "__main__":
    main()
