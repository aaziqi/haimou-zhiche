import argparse
import json
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_project_root(repo_root: Path) -> Path:
    for candidate in [repo_root.parents[1] if len(repo_root.parents) > 1 else repo_root, repo_root.parent, repo_root]:
        if (candidate / "datasets").exists() or (candidate / "results").exists():
            return candidate
    return repo_root


def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def pick_files(files: list[Path], n: int, *, seed: int) -> list[Path]:
    if n <= 0 or not files:
        return []
    rng = random.Random(seed)
    if n >= len(files):
        out = list(files)
        rng.shuffle(out)
        return out
    out = rng.sample(files, n)
    out.sort(key=lambda p: p.name)
    return out


def copy_many(pairs: list[tuple[Path, Path]]):
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def extract_names_line(data_yaml: Path) -> tuple[str, int | None]:
    text = data_yaml.read_text(encoding="utf-8", errors="ignore")
    nc = None
    for line in text.splitlines():
        m = re.match(r"\s*nc\s*:\s*(\d+)\s*$", line)
        if m:
            nc = int(m.group(1))
            break
    names_line = ""
    for line in text.splitlines():
        if re.match(r"\s*names\s*:\s*", line):
            names_line = line.strip()
            break
    return names_line, nc


def write_subset_data_yaml(out_root: Path, *, source_data_yaml: Path):
    names_line, nc = extract_names_line(source_data_yaml)
    out_path_value = str(out_root).replace("\\", "/")
    lines = [
        f"path: {out_path_value}",
        "train: images/train",
        "val: images/val",
    ]
    if nc is not None:
        lines.append(f"nc: {nc}")
    if names_line:
        lines.append(names_line)
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_yolo_subset(
    *,
    yolo_root: Path,
    out_root: Path,
    n_train: int,
    n_val: int,
    seed: int,
):
    img_train = list_images(yolo_root / "images" / "train")
    img_val = list_images(yolo_root / "images" / "val")
    pick_train = pick_files(img_train, n_train, seed=seed)
    pick_val = pick_files(img_val, n_val, seed=seed + 1)

    pairs: list[tuple[Path, Path]] = []
    for split, images in [("train", pick_train), ("val", pick_val)]:
        for p in images:
            pairs.append((p, out_root / "images" / split / p.name))
            label = (yolo_root / "labels" / split / f"{p.stem}.txt")
            if label.exists():
                pairs.append((label, out_root / "labels" / split / label.name))
    copy_many(pairs)
    if (yolo_root / "data.yaml").exists():
        write_subset_data_yaml(out_root, source_data_yaml=yolo_root / "data.yaml")

    return {
        "source": str(yolo_root),
        "out": str(out_root),
        "train_images": len(pick_train),
        "val_images": len(pick_val),
    }


def build_unpaired_subset(
    *,
    unpaired_root: Path,
    out_root: Path,
    n_a: int,
    n_b: int,
    seed: int,
):
    a = list_images(unpaired_root / "trainA")
    b = list_images(unpaired_root / "trainB")
    pick_a = pick_files(a, n_a, seed=seed)
    pick_b = pick_files(b, n_b, seed=seed + 7)
    pairs: list[tuple[Path, Path]] = []
    for p in pick_a:
        pairs.append((p, out_root / "trainA" / p.name))
    for p in pick_b:
        pairs.append((p, out_root / "trainB" / p.name))
    copy_many(pairs)
    return {
        "source": str(unpaired_root),
        "out": str(out_root),
        "trainA": len(pick_a),
        "trainB": len(pick_b),
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    project_root = resolve_project_root(repo_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(project_root / "quick_check_samples"))
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--yolo_root", default=str(project_root / "datasets" / "URPC_optical" / "yolo_new"))
    ap.add_argument("--yolo_train", type=int, default=80)
    ap.add_argument("--yolo_val", type=int, default=40)

    ap.add_argument("--euvp_root", default=str(project_root / "EUVP_Unpaired"))
    ap.add_argument("--euvp_a", type=int, default=40)
    ap.add_argument("--euvp_b", type=int, default=40)

    ap.add_argument("--uieb_root", default=str(project_root / "UIEB_Unpaired"))
    ap.add_argument("--uieb_a", type=int, default=40)
    ap.add_argument("--uieb_b", type=int, default=40)

    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    ensure_empty_dir(out_root)

    manifest: dict = {
        "project_root": str(project_root),
        "out_root": str(out_root),
        "seed": int(args.seed),
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "items": {},
    }

    yolo_root = Path(args.yolo_root).resolve()
    if yolo_root.exists():
        info = build_yolo_subset(
            yolo_root=yolo_root,
            out_root=out_root / "yolo_urpc_subset",
            n_train=int(args.yolo_train),
            n_val=int(args.yolo_val),
            seed=int(args.seed),
        )
        manifest["items"]["yolo_urpc_subset"] = info

    euvp_root = Path(args.euvp_root).resolve()
    if euvp_root.exists():
        info = build_unpaired_subset(
            unpaired_root=euvp_root,
            out_root=out_root / "enhance_euvp_subset",
            n_a=int(args.euvp_a),
            n_b=int(args.euvp_b),
            seed=int(args.seed) + 100,
        )
        manifest["items"]["enhance_euvp_subset"] = info

    uieb_root = Path(args.uieb_root).resolve()
    if uieb_root.exists():
        info = build_unpaired_subset(
            unpaired_root=uieb_root,
            out_root=out_root / "enhance_uieb_subset",
            n_a=int(args.uieb_a),
            n_b=int(args.uieb_b),
            seed=int(args.seed) + 200,
        )
        manifest["items"]["enhance_uieb_subset"] = info

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", out_root)
    print(json.dumps(manifest["items"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
