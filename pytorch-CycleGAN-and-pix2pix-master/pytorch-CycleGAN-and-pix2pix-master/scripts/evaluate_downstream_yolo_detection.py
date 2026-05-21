import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path, env: dict | None = None):
    print("Running:", " ".join(str(x) for x in cmd))
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _collect_images(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _enhance_folder(
    *,
    inp_dir: Path,
    out_root: Path,
    cyclegan_name: str,
    epoch: str,
    results_dir: Path,
    device: str,
    num_test: int | None,
):
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = env.get("KMP_DUPLICATE_LIB_OK", "TRUE")
    if device.lower() == "cpu":
        env["CYCLEGAN_DEVICE"] = "cpu"

    ckpt_dir = REPO_ROOT / "checkpoints" / cyclegan_name
    model_suffix = ""
    if ckpt_dir.exists():
        epoch_s = str(epoch)
        if not (ckpt_dir / f"{epoch_s}_net_G.pth").exists() and not (ckpt_dir / "latest_net_G.pth").exists():
            if (ckpt_dir / f"{epoch_s}_net_G_A.pth").exists() or (ckpt_dir / "latest_net_G_A.pth").exists():
                model_suffix = "_A"

    cmd = [
        sys.executable,
        "test.py",
        "--dataroot",
        str(inp_dir),
        "--name",
        cyclegan_name,
        "--model",
        "test",
        "--no_dropout",
        "--results_dir",
        str(results_dir),
        "--epoch",
        str(epoch),
    ]
    if model_suffix:
        cmd += ["--model_suffix", model_suffix]
    if num_test is not None and num_test > 0:
        cmd += ["--num_test", str(num_test)]
    _run(cmd, cwd=REPO_ROOT, env=env)

    images_dir = results_dir / cyclegan_name / f"test_{epoch}" / "images"
    if not images_dir.exists():
        images_dir = results_dir / cyclegan_name / f"test_{epoch}_iter0" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Cannot find CycleGAN test output folder under {results_dir}")

    fake_b = sorted(images_dir.glob("*_fake_B.png"))
    suffix = "_fake_B.png"
    if not fake_b:
        fake_b = sorted(images_dir.glob("*_fake.png"))
        suffix = "_fake.png"
    if not fake_b:
        fake_b = sorted(images_dir.glob("*_fake_B.jpg"))
        suffix = "_fake_B.jpg"
    if not fake_b:
        fake_b = sorted(images_dir.glob("*_fake.jpg"))
        suffix = "_fake.jpg"
    if not fake_b:
        raise FileNotFoundError(f"No enhanced images found in {images_dir}")

    _ensure_empty_dir(out_root)
    for p in fake_b:
        stem = p.name[: -len(suffix)]
        shutil.copy2(p, out_root / f"{stem}.png")


def _write_data_yaml(base: dict, out_yaml: Path, *, val_images_dir: Path, labels_dir: Path | None):
    d = dict(base)
    d["val"] = str(val_images_dir)
    if labels_dir is not None:
        d["path"] = str(labels_dir.parent.parent) if "path" not in d else d["path"]
    out_yaml.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


def _resolve_images_dir(data_yaml: Path, base: dict, key: str) -> Path:
    p = Path(base[key])
    if not p.is_absolute():
        p = (data_yaml.parent / p).resolve()
    return p


def _guess_labels_dir(images_dir: Path, data_yaml: Path, base: dict, split: str) -> Path | None:
    labels_guess = None
    if "path" in base:
        base_path = Path(base["path"])
        if not base_path.is_absolute():
            base_path = (data_yaml.parent / base_path).resolve()
        labels_guess = base_path / "labels" / split
    else:
        s = str(images_dir).replace("\\", "/")
        if s.endswith(f"/images/{split}") or (split == "val" and s.endswith("/images/valid")):
            labels_guess = Path(s.replace(f"images/{split}", f"labels/{split}").replace(f"images\\{split}", f"labels\\{split}")).resolve()
    return labels_guess if labels_guess is not None and labels_guess.exists() else None


def main():
    print("Starting downstream evaluation...")
    os.environ["KMP_DUPLICATE_LIB_OK"] = os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO data YAML (train/val paths and names).")
    ap.add_argument("--yolo_model", default="yolov8n.pt", help="Ultralytics model or weights path.")
    ap.add_argument("--yolo_epochs", type=int, default=0, help="If >0, train detector before evaluation.")
    ap.add_argument("--yolo_seed", type=int, default=0, help="Ultralytics training seed (only when yolo_epochs>0).")
    ap.add_argument("--yolo_name", default="", help="Ultralytics run name (only when yolo_epochs>0).")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--cyclegan_name", required=True, help="CycleGAN checkpoint folder name.")
    ap.add_argument("--cyclegan_epoch", default="latest")
    ap.add_argument("--num_test", type=int, default=0, help="Number of val images to enhance (0 means all).")
    ap.add_argument("--eval_enh_data", default="", help="Optional: prebuilt enhanced-domain YOLO data.yaml for evaluation. If set, skips enhancement generation.")
    ap.add_argument("--train_domain", default="raw", choices=["raw", "enhanced", "mixed"])
    ap.add_argument("--train_on", default="raw", choices=["raw", "enhanced"], help="Which domain to use for training/val selection when yolo_epochs>0.")
    ap.add_argument("--num_train", type=int, default=0, help="Number of train images to enhance when train_domain!=raw (0 means all).")
    ap.add_argument("--train_data", default="", help="Optional: override YOLO data.yaml used for detector training (only when yolo_epochs>0).")
    ap.add_argument("--workdir", default=str(REPO_ROOT / "results" / "downstream_yolo"))
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    (workdir / "ultralytics_config").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = os.environ.get("YOLO_CONFIG_DIR", str(workdir / "ultralytics_config"))

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Ultralytics is required. Install it with: pip install ultralytics", file=sys.stderr)
        raise e

    data_yaml = Path(args.data).expanduser().resolve()
    base = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    workdir.mkdir(parents=True, exist_ok=True)

    val_images = _resolve_images_dir(data_yaml, base, "val")
    if val_images.is_dir():
        imgs = _collect_images(val_images)
        if not imgs:
            raise FileNotFoundError(f"No images found under val path: {val_images}")
    else:
        raise FileNotFoundError(f"val path is not a directory: {val_images}")
    val_count_raw = len(_collect_images(val_images))

    train_images = _resolve_images_dir(data_yaml, base, "train")
    if not train_images.exists():
        raise FileNotFoundError(f"train path does not exist: {train_images}")
    train_count_raw = len(_collect_images(train_images)) if train_images.is_dir() else 0

    results_dir = workdir / "cyclegan_results"
    if args.eval_enh_data:
        enhanced_yaml = Path(args.eval_enh_data).expanduser().resolve()
        if not enhanced_yaml.exists():
            raise FileNotFoundError(f"eval_enh_data not found: {enhanced_yaml}")
    else:
        enhanced_images_dir = workdir / "enhanced_val_images"
        expected_val_enh = val_count_raw if args.num_test == 0 else args.num_test
        cached_val_enh = len(list(enhanced_images_dir.glob("*.png"))) if enhanced_images_dir.exists() else 0
        if not (expected_val_enh > 0 and cached_val_enh >= expected_val_enh):
            _enhance_folder(
                inp_dir=val_images,
                out_root=enhanced_images_dir,
                cyclegan_name=args.cyclegan_name,
                epoch=args.cyclegan_epoch,
                results_dir=results_dir,
                device=args.device,
                num_test=args.num_test if args.num_test > 0 else val_count_raw,
            )

        enhanced_dataset_dir = workdir / "enhanced_dataset"
        enhanced_labels_dir = enhanced_dataset_dir / "labels" / "val"
        enhanced_images_root = enhanced_dataset_dir / "images" / "val"
        _ensure_empty_dir(enhanced_images_root)
        _ensure_empty_dir(enhanced_labels_dir)

        for p in enhanced_images_dir.glob("*.png"):
            shutil.copy2(p, enhanced_images_root / p.name)

        labels_guess = _guess_labels_dir(val_images, data_yaml, base, "val")
        if labels_guess is None:
            print("Warning: cannot locate labels/val automatically. Enhanced evaluation may fail if labels are not discoverable.")
        else:
            for p in labels_guess.glob("*.txt"):
                shutil.copy2(p, enhanced_labels_dir / p.name)

        enhanced_yaml = workdir / "data_enhanced.yaml"
        _write_data_yaml(base, enhanced_yaml, val_images_dir=enhanced_images_root, labels_dir=enhanced_labels_dir if enhanced_labels_dir.exists() else None)
        (workdir / "stage_enhance_done.txt").write_text("ok\n", encoding="utf-8")

    train_domain_yaml = data_yaml
    train_domain_tag = "raw"
    if args.train_domain != "raw":
        train_labels = _guess_labels_dir(train_images, data_yaml, base, "train")
        if train_labels is None:
            raise FileNotFoundError("Cannot locate labels/train automatically. Please ensure data.yaml has 'path: ...' or standard images/train layout.")

        enhanced_train_images_dir = workdir / "enhanced_train_images"
        expected_train_enh = train_count_raw if args.num_train == 0 else args.num_train
        cached_train_enh = len(list(enhanced_train_images_dir.glob("*.png"))) if enhanced_train_images_dir.exists() else 0
        if not (expected_train_enh > 0 and cached_train_enh >= expected_train_enh):
            _enhance_folder(
                inp_dir=train_images,
                out_root=enhanced_train_images_dir,
                cyclegan_name=args.cyclegan_name,
                epoch=args.cyclegan_epoch,
                results_dir=results_dir,
                device=args.device,
                num_test=args.num_train if args.num_train > 0 else train_count_raw,
            )

        enhanced_train_dataset_dir = workdir / "enhanced_train_dataset"
        enh_train_labels_dir = enhanced_train_dataset_dir / "labels" / "train"
        enh_train_images_root = enhanced_train_dataset_dir / "images" / "train"
        print("Preparing enhanced train dataset...")
        _ensure_empty_dir(enh_train_images_root)
        _ensure_empty_dir(enh_train_labels_dir)

        for p in enhanced_train_images_dir.glob("*.png"):
            shutil.copy2(p, enh_train_images_root / p.name)
            src = train_labels / f"{p.stem}.txt"
            if src.exists():
                shutil.copy2(src, enh_train_labels_dir / f"{p.stem}.txt")
        print("Enhanced train images:", len(list(enh_train_images_root.glob("*.png"))))

        if args.train_domain == "enhanced":
            train_domain_tag = "enhanced"
            train_yaml = workdir / "data_train_enhanced.yaml"
            d = dict(base)
            d["train"] = str(enh_train_images_root)
            d["val"] = str(enhanced_images_root if args.train_on == "enhanced" else val_images)
            d["path"] = str(enhanced_train_dataset_dir) if "path" not in d else d["path"]
            train_yaml.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
            train_domain_yaml = train_yaml
        elif args.train_domain == "mixed":
            train_domain_tag = "mixed"
            mixed_dir = workdir / "mixed_train_dataset"
            mixed_images = mixed_dir / "images" / "train"
            mixed_labels = mixed_dir / "labels" / "train"
            _ensure_empty_dir(mixed_images)
            _ensure_empty_dir(mixed_labels)

            for p in _collect_images(train_images):
                shutil.copy2(p, mixed_images / p.name)
                src = train_labels / f"{p.stem}.txt"
                if src.exists():
                    shutil.copy2(src, mixed_labels / f"{p.stem}.txt")

            for p in enh_train_images_root.glob("*.png"):
                out_name = f"{p.stem}_enh.png"
                shutil.copy2(p, mixed_images / out_name)
                src = enh_train_labels_dir / f"{p.stem}.txt"
                if src.exists():
                    shutil.copy2(src, mixed_labels / f"{p.stem}_enh.txt")

            train_yaml = workdir / "data_train_mixed.yaml"
            d = dict(base)
            d["train"] = str(mixed_images)
            d["val"] = str(enhanced_images_root if args.train_on == "enhanced" else val_images)
            d["path"] = str(mixed_dir) if "path" not in d else d["path"]
            train_yaml.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
            train_domain_yaml = train_yaml

    if args.train_data:
        train_domain_yaml = Path(args.train_data).expanduser().resolve()
        if not train_domain_yaml.exists():
            raise FileNotFoundError(f"train_data not found: {train_domain_yaml}")

    model = YOLO(args.yolo_model)
    yolo_run_dir = None
    yolo_best = None
    yolo_log = workdir / "yolo_train_val.log"
    if args.yolo_epochs and args.yolo_epochs > 0:
        run_name = args.yolo_name.strip() or f"{train_domain_tag}_s{int(args.yolo_seed)}"
        with yolo_log.open("w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
            model.train(
                data=str(train_domain_yaml),
                epochs=int(args.yolo_epochs),
                imgsz=int(args.imgsz),
                device=args.device,
                seed=int(args.yolo_seed),
                name=run_name,
                verbose=False,
                plots=False,
                workers=0,
            )
        yolo_run_dir = str(getattr(model, "trainer", None).save_dir) if getattr(model, "trainer", None) is not None else None
        best = Path(yolo_run_dir) / "weights" / "best.pt" if yolo_run_dir else None
        if best is not None and best.exists():
            yolo_best = str(best)
            model = YOLO(yolo_best)

    with yolo_log.open("a", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
        metrics_raw = model.val(data=str(data_yaml), imgsz=int(args.imgsz), device=args.device, verbose=False, plots=False, workers=0)
    (workdir / "stage_raw_val_done.txt").write_text("ok\n", encoding="utf-8")
    with yolo_log.open("a", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
        metrics_enh = model.val(data=str(enhanced_yaml), imgsz=int(args.imgsz), device=args.device, verbose=False, plots=False, workers=0)
    (workdir / "stage_enh_val_done.txt").write_text("ok\n", encoding="utf-8")

    out = {
        "data_raw": str(data_yaml),
        "data_enhanced": str(enhanced_yaml),
        "cyclegan_name": args.cyclegan_name,
        "cyclegan_epoch": args.cyclegan_epoch,
        "train_domain": train_domain_tag,
        "train_data": str(train_domain_yaml),
        "yolo_model": args.yolo_model,
        "yolo_run_dir": yolo_run_dir,
        "yolo_best": yolo_best,
        "yolo_epochs": args.yolo_epochs,
        "yolo_seed": int(args.yolo_seed),
        "yolo_name": args.yolo_name.strip() or f"{train_domain_tag}_s{int(args.yolo_seed)}",
        "imgsz": args.imgsz,
        "device": args.device,
        "val_count": val_count_raw,
        "train_count": train_count_raw,
        "metrics_raw": getattr(metrics_raw, "results_dict", None) or {},
        "metrics_enhanced": getattr(metrics_enh, "results_dict", None) or {},
    }
    (workdir / "downstream_detection_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        import traceback
        with open("error_yolo.log", "w") as f:
            traceback.print_exc(file=f)
        print("EXCEPTION:", e, file=sys.stderr)
        raise
