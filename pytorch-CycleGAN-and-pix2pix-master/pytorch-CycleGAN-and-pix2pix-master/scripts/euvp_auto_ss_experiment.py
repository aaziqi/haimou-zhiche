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


def _cmd_to_str(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


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

    def _name_matches(n: str, patterns: list[str]) -> bool:
        name = n.lower()
        for p in patterns:
            pl = p.lower()
            if name == pl:
                return True
            if name.startswith(pl + "-") or name.startswith(pl + "_"):
                return True
            if pl in {"raw", "reference"} and name.startswith(pl):
                return True
        return False

    def _dir_has_images(d: Path, max_entries: int = 2000) -> bool:
        try:
            for i, p in enumerate(d.iterdir()):
                if i >= max_entries:
                    break
                if p.is_file() and _is_image_file(p):
                    return True
        except FileNotFoundError:
            return False
        return False

    def _best_image_dir(d: Path, max_depth: int = 8) -> Path | None:
        counts: dict[Path, int] = {}
        try:
            for p in d.rglob("*"):
                if not p.is_file():
                    continue
                if not _is_image_file(p):
                    continue
                try:
                    rel = p.relative_to(d)
                except ValueError:
                    continue
                if len(rel.parts) - 1 > max_depth:
                    continue
                parent = p.parent
                counts[parent] = counts.get(parent, 0) + 1
        except FileNotFoundError:
            return None
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: (kv[1], -len(kv[0].relative_to(d).parts)))[0]

    try:
        subdirs = [p for p in root.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return None, None

    for d in subdirs:
        inp = _first_existing_dir([d / n for n in inp_names])
        ref = _first_existing_dir([d / n for n in ref_names])
        if inp is not None:
            return inp, ref

    best_inp = None
    best_ref = None
    try:
        for p in root.rglob("*"):
            if not p.is_dir():
                continue
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            if len(rel.parts) > 8:
                continue
            if not _dir_has_images(p):
                continue
            if best_inp is None and _name_matches(p.name, inp_names):
                best_inp = p
            if best_ref is None and _name_matches(p.name, ref_names):
                best_ref = p
            if best_inp is not None and best_ref is not None:
                break
    except FileNotFoundError:
        pass

    if best_inp is None:
        best_inp = _best_image_dir(root)
    if best_ref is None and best_inp is not None:
        sib = _first_existing_dir([best_inp.parent / n for n in ref_names])
        if sib is not None and _dir_has_images(sib):
            best_ref = sib

    return best_inp, best_ref


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


def generate_baseline_outputs(inp_dir: Path, out_dir: Path, method: str, max_images: int, gamma: float, clahe_clip: float, clahe_tile: tuple[int, int]):
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
            out = _clahe_lab(img, clip_limit=float(clahe_clip), tile_grid_size=clahe_tile)
        elif method == "gamma":
            out = _gamma(img, gamma=float(gamma))
        elif method == "grayworld_clahe":
            out = _clahe_lab(_gray_world_wb(img), clip_limit=float(clahe_clip), tile_grid_size=clahe_tile)
        else:
            raise ValueError(f"Unknown baseline method: {method}")
        _write_png(out_dir / f"{p.stem}__fake.png", out)


def run_test_single(name: str, epoch: str, inp_dir: Path, num_test: int, model_suffix: str, gpu_ids: str, results_dir: Path | None = None):
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
    if results_dir is not None:
        cmd += ["--results_dir", str(results_dir)]
    print("Running:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return True


def _paper_preset_entries():
    entries = []
    entries += [
        {"group": "paper:table1_euvp", "dataset": "euvp", "kind": "baseline", "name": "identity", "epoch": ""},
        {"group": "paper:table1_euvp", "dataset": "euvp", "kind": "baseline", "name": "grayworld", "epoch": ""},
        {"group": "paper:table1_euvp", "dataset": "euvp", "kind": "baseline", "name": "clahe", "epoch": ""},
        {"group": "paper:table1_euvp", "dataset": "euvp", "kind": "baseline", "name": "grayworld_clahe", "epoch": ""},
        {"group": "paper:table1_euvp", "dataset": "euvp", "kind": "model", "name": "euvp_cyclegan_full", "epoch": "200"},
        {"group": "paper:table1_euvp", "dataset": "euvp", "kind": "model", "name": "euvp_mpcgan_stage2_s0", "epoch": "202"},
    ]

    entries += [
        {"group": "paper:table4_uieb", "dataset": "uieb", "kind": "baseline", "name": "identity", "epoch": ""},
        {"group": "paper:table4_uieb", "dataset": "uieb", "kind": "baseline", "name": "grayworld", "epoch": ""},
        {"group": "paper:table4_uieb", "dataset": "uieb", "kind": "baseline", "name": "clahe", "epoch": ""},
        {"group": "paper:table4_uieb", "dataset": "uieb", "kind": "baseline", "name": "grayworld_clahe", "epoch": ""},
        {"group": "paper:table4_uieb", "dataset": "uieb", "kind": "model", "name": "euvp_cyclegan_full", "epoch": "200"},
        {"group": "paper:table4_uieb", "dataset": "uieb", "kind": "model", "name": "euvp_mpcgan_stage2_s0", "epoch": "202"},
    ]

    entries += [
        {"group": "paper:table2_ablation", "dataset": "euvp", "kind": "model", "name": "abl_uieb_A", "epoch": "20"},
        {"group": "paper:table2_ablation", "dataset": "euvp", "kind": "model", "name": "abl_uieb_B", "epoch": "20"},
        {"group": "paper:table2_ablation", "dataset": "euvp", "kind": "model", "name": "abl_uieb_C", "epoch": "20"},
        {"group": "paper:table2_ablation", "dataset": "euvp", "kind": "model", "name": "abl_uieb_D", "epoch": "20"},
        {"group": "paper:table2_ablation", "dataset": "euvp", "kind": "model", "name": "abl_uieb_E", "epoch": "20"},
    ]

    entries += [
        {"group": "paper:external_sota", "dataset": "euvp", "kind": "external", "name": "WaterNet", "epoch": ""},
        {"group": "paper:external_sota", "dataset": "euvp", "kind": "external", "name": "UColor", "epoch": ""},
        {"group": "paper:external_sota", "dataset": "euvp", "kind": "external", "name": "FUnIE-GAN", "epoch": ""},
    ]
    return entries


def _matrix_rows_from_entries(entries, dataset_cfg: dict, bench_dir: Path, args):
    rows = []
    for e in entries:
        dname = str(e.get("dataset", "")).strip().lower()
        kind = str(e.get("kind", "")).strip().lower()
        name = str(e.get("name", "")).strip()
        epoch = str(e.get("epoch", "")).strip()
        group = str(e.get("group", "")).strip() or "custom"

        cfg = dataset_cfg.get(dname) or {}
        inp_dir = cfg.get("inp")
        ref_dir = cfg.get("ref")

        if kind == "baseline":
            out_dir = bench_dir / dname / "baselines" / name / "images"
            per_csv = bench_dir / dname / "baselines" / f"{name}_per_image.csv"
            per_json = bench_dir / dname / "baselines" / f"{name}_summary.json"
            eval_cmd = [
                sys.executable,
                "scripts/evaluate_euvp_psnr_ssim.py",
                "--inp_dir",
                str(inp_dir) if inp_dir is not None else "",
                "--pred_dir",
                str(out_dir),
                "--bootstrap_iters",
                str(args.bootstrap_iters),
                "--seed",
                str(args.seed),
            ]
            if args.num_test and int(args.num_test) > 0:
                eval_cmd += ["--max_images", str(args.num_test)]
            if ref_dir is not None:
                eval_cmd += ["--ref_dir", str(ref_dir)]
            if args.report_uiqm_parts:
                eval_cmd += ["--report_uiqm_parts"]
            eval_cmd += ["--save_csv", str(per_csv), "--save_json", str(per_json)]
            rows.append(
                {
                    "group": group,
                    "dataset": dname,
                    "kind": kind,
                    "name": name,
                    "epoch": "",
                    "pred_dir": str(out_dir),
                    "run_cmd": _cmd_to_str(
                        [
                            sys.executable,
                            "scripts/euvp_auto_ss_experiment.py",
                            "--datasets",
                            dname,
                            "--skip_models",
                            "--include_baselines",
                            "--baseline_methods",
                            name,
                            "--num_test",
                            str(args.num_test),
                            "--bootstrap_iters",
                            str(args.bootstrap_iters),
                            "--seed",
                            str(args.seed),
                        ]
                    ),
                    "inference_cmd": "",
                    "eval_cmd": _cmd_to_str(eval_cmd),
                    "per_image_csv": str(per_csv),
                    "summary_json": str(per_json),
                    "paper_targets": group,
                    "notes": "internal_baseline",
                }
            )
            continue

        if kind == "model":
            dataset_results_dir = REPO_ROOT / "results" / dname
            pred_dir = dataset_results_dir / name / f"test_{epoch}" / "images"
            per_dir = bench_dir / dname / "models" / name / f"epoch_{epoch}"
            per_csv = per_dir / "per_image.csv"
            per_json = per_dir / "summary.json"
            inference_cmd = [
                sys.executable,
                "test.py",
                "--dataroot",
                str(inp_dir) if inp_dir is not None else "",
                "--name",
                name,
                "--model",
                "test",
                "--dataset_mode",
                "single",
                "--model_suffix",
                str(args.model_suffix),
                "--epoch",
                str(epoch),
                "--num_test",
                str(args.num_test),
                "--results_dir",
                str(dataset_results_dir),
            ]
            eval_cmd = [
                sys.executable,
                "scripts/evaluate_euvp_psnr_ssim.py",
                "--inp_dir",
                str(inp_dir) if inp_dir is not None else "",
                "--pred_dir",
                str(pred_dir),
                "--bootstrap_iters",
                str(args.bootstrap_iters),
                "--seed",
                str(args.seed),
            ]
            if args.num_test and int(args.num_test) > 0:
                eval_cmd += ["--max_images", str(args.num_test)]
            if ref_dir is not None:
                eval_cmd += ["--ref_dir", str(ref_dir)]
            if args.report_uiqm_parts:
                eval_cmd += ["--report_uiqm_parts"]
            eval_cmd += ["--save_csv", str(per_csv), "--save_json", str(per_json)]
            rows.append(
                {
                    "group": group,
                    "dataset": dname,
                    "kind": kind,
                    "name": name,
                    "epoch": epoch,
                    "pred_dir": str(pred_dir),
                    "run_cmd": _cmd_to_str(
                        [
                            sys.executable,
                            "scripts/euvp_auto_ss_experiment.py",
                            "--datasets",
                            dname,
                            "--models",
                            name,
                            "--epochs",
                            str(epoch),
                            "--run_inference",
                            "--skip_baselines",
                            "--num_test",
                            str(args.num_test),
                            "--bootstrap_iters",
                            str(args.bootstrap_iters),
                            "--seed",
                            str(args.seed),
                            "--gpu_ids",
                            str(args.gpu_ids),
                        ]
                    ),
                    "inference_cmd": _cmd_to_str(inference_cmd),
                    "eval_cmd": _cmd_to_str(eval_cmd),
                    "per_image_csv": str(per_csv),
                    "summary_json": str(per_json),
                    "paper_targets": group,
                    "notes": "internal_model",
                }
            )
            continue

        if kind == "external":
            ext_pred_dir = REPO_ROOT / "results" / "external" / name / dname / "images"
            per_dir = bench_dir / dname / "external" / name
            per_csv = per_dir / "per_image.csv"
            per_json = per_dir / "summary.json"
            eval_cmd = [
                sys.executable,
                "scripts/evaluate_euvp_psnr_ssim.py",
                "--inp_dir",
                str(inp_dir) if inp_dir is not None else "",
                "--pred_dir",
                str(ext_pred_dir),
                "--bootstrap_iters",
                str(args.bootstrap_iters),
                "--seed",
                str(args.seed),
            ]
            if args.num_test and int(args.num_test) > 0:
                eval_cmd += ["--max_images", str(args.num_test)]
            if ref_dir is not None:
                eval_cmd += ["--ref_dir", str(ref_dir)]
            if args.report_uiqm_parts:
                eval_cmd += ["--report_uiqm_parts"]
            eval_cmd += ["--save_csv", str(per_csv), "--save_json", str(per_json)]
            rows.append(
                {
                    "group": group,
                    "dataset": dname,
                    "kind": kind,
                    "name": name,
                    "epoch": "",
                    "pred_dir": str(ext_pred_dir),
                    "run_cmd": "",
                    "inference_cmd": "",
                    "eval_cmd": _cmd_to_str(eval_cmd),
                    "per_image_csv": str(per_csv),
                    "summary_json": str(per_json),
                    "paper_targets": group,
                    "notes": "place predicted images at pred_dir; filenames should share stem with inputs; prefer *_fake.png",
                }
            )
            continue

        rows.append(
            {
                "group": group,
                "dataset": dname,
                "kind": kind,
                "name": name,
                "epoch": epoch,
                "pred_dir": "",
                "run_cmd": "",
                "inference_cmd": "",
                "eval_cmd": "",
                "per_image_csv": "",
                "summary_json": "",
                "paper_targets": group,
                "notes": "unknown_kind",
            }
        )
    return rows


def _write_matrix_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "dataset",
        "kind",
        "name",
        "epoch",
        "pred_dir",
        "run_cmd",
        "inference_cmd",
        "eval_cmd",
        "per_image_csv",
        "summary_json",
        "paper_targets",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


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
    parser.add_argument("--preset", default="")
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
    parser.add_argument("--skip_models", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--baseline_methods", default="identity,grayworld,clahe,gamma,grayworld_clahe")
    parser.add_argument("--baseline_gamma", type=float, default=1.2)
    parser.add_argument("--baseline_clahe_clip", type=float, default=2.0)
    parser.add_argument("--baseline_clahe_tile", default="8,8")
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--report_uiqm_parts", action="store_true")
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--write_matrix", action="store_true")
    parser.add_argument("--matrix_out", default="")
    parser.add_argument("--make_figures", action="store_true")
    args = parser.parse_args()

    if str(args.preset).strip().lower() == "paper":
        args.datasets = "euvp,uieb"
        args.include_baselines = True
        args.models = "euvp_cyclegan_full,euvp_mpcgan_stage2_s0,abl_uieb_A,abl_uieb_B,abl_uieb_C,abl_uieb_D,abl_uieb_E"
        args.epochs = "200,202,20"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    epochs = [e.strip() for e in args.epochs.split(",") if e.strip()]
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    baseline_methods = [x.strip() for x in str(args.baseline_methods).split(",") if x.strip()]
    tile_parts = [x.strip() for x in str(args.baseline_clahe_tile).split(",") if x.strip()]
    if len(tile_parts) == 2:
        try:
            clahe_tile = (int(tile_parts[0]), int(tile_parts[1]))
        except ValueError:
            clahe_tile = (8, 8)
    else:
        clahe_tile = (8, 8)

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

    entries = _paper_preset_entries() if str(args.preset).strip().lower() == "paper" else []
    if not entries:
        if args.include_baselines and not args.skip_baselines:
            for dname in dataset_cfg.keys():
                for method in baseline_methods:
                    entries.append({"group": "custom:baselines", "dataset": dname, "kind": "baseline", "name": method, "epoch": ""})
        if models and epochs and not args.skip_models:
            for dname in dataset_cfg.keys():
                for m in models:
                    for ep in epochs:
                        entries.append({"group": "custom:models", "dataset": dname, "kind": "model", "name": m, "epoch": ep})

    matrix_rows = _matrix_rows_from_entries(entries, dataset_cfg, bench_dir, args)
    matrix_path = Path(args.matrix_out) if str(args.matrix_out).strip() else (bench_dir / "experiment_matrix.csv")
    if args.write_matrix:
        _write_matrix_csv(matrix_path, matrix_rows)
        print("Saved experiment matrix to", matrix_path)
    if args.plan_only:
        for r in matrix_rows:
            print(f"[{r.get('group')}] {r.get('dataset')} {r.get('kind')} {r.get('name')} {r.get('epoch')}".strip())
            if r.get("run_cmd"):
                print("  run_cmd:", r.get("run_cmd"))
            if r.get("inference_cmd"):
                print("  inference_cmd:", r.get("inference_cmd"))
            if r.get("eval_cmd"):
                print("  eval_cmd:", r.get("eval_cmd"))
            if r.get("summary_json"):
                print("  summary_json:", r.get("summary_json"))
        return

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

        if args.include_baselines and not args.skip_baselines:
            for method in baseline_methods:
                out_dir = bench_dir / dname / "baselines" / method / "images"
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                generate_baseline_outputs(inp_dir, out_dir, method, args.num_test, args.baseline_gamma, args.baseline_clahe_clip, clahe_tile)
                per_csv = bench_dir / dname / "baselines" / f"{method}_per_image.csv"
                per_json = bench_dir / dname / "baselines" / f"{method}_summary.json"
                _, metrics = run_evaluate(inp_dir, ref_dir, out_dir, per_csv, per_json, args.bootstrap_iters, args.seed, args.num_test, args.report_uiqm_parts)
                metrics.update({"dataset": dname, "model": f"baseline:{method}", "epoch": ""})
                summary_rows.append(metrics)

        if not args.skip_models:
            for model in models:
                for epoch in epochs:
                    dataset_results_dir = REPO_ROOT / "results" / dname
                    pred_dir = dataset_results_dir / model / f"test_{epoch}" / "images"
                    if args.run_inference or not pred_dir.exists():
                        ok = run_test_single(model, epoch, inp_dir, args.num_test, args.model_suffix, args.gpu_ids, dataset_results_dir)
                        if not ok:
                            continue
                    per_dir = bench_dir / dname / "models" / model / f"epoch_{epoch}"
                    per_csv = per_dir / "per_image.csv"
                    per_json = per_dir / "summary.json"
                    _, metrics = run_evaluate(inp_dir, ref_dir, pred_dir, per_csv, per_json, args.bootstrap_iters, args.seed, args.num_test, args.report_uiqm_parts)
                    metrics.update({"dataset": dname, "model": model, "epoch": epoch})
                    summary_rows.append(metrics)

    out_csv = bench_dir / "summary.csv"
    all_rows = _iter_existing_benchmark_summaries(bench_dir)
    if not all_rows and summary_rows:
        all_rows = summary_rows
    if not all_rows:
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
        for r in sorted(all_rows, key=lambda x: (str(x.get("dataset", "")), str(x.get("model", "")), str(x.get("epoch", "")))):
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print("Saved summary to", out_csv)

    if args.make_figures:
        run([sys.executable, "scripts/plot_unified_paper_figures.py"])
        run([sys.executable, "scripts/plot_paper_extras.py"])


if __name__ == "__main__":
    main()
