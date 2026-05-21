import argparse
import shutil
from pathlib import Path

import yaml


def _ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _resolve_from_yaml(data_yaml: Path, base: dict, key: str) -> Path:
    p = Path(base[key])
    if not p.is_absolute():
        p = (data_yaml.parent / p).resolve()
    return p


def _resolve_root(data_yaml: Path, base: dict, train_images: Path) -> Path:
    if "path" in base:
        root = Path(base["path"])
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return train_images.parent.parent


def _load_names(data_yaml: Path) -> list[str]:
    base = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = base.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
    return [str(x) for x in names]


def _write_yaml(out_yaml: Path, *, root: Path, train_images: Path, val_images: Path, names: list[str]):
    d = {
        "path": str(root),
        "train": str(train_images),
        "val": str(val_images),
        "nc": len(names),
        "names": names,
    }
    out_yaml.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_data", required=True, help="Original YOLO data.yaml (raw train/val + labels).")
    ap.add_argument("--enh_root", required=True, help="Enhanced dataset root (must contain images/train, labels/train, images/val, labels/val).")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--raw_train_limit", type=int, default=0, help="If >0, limit number of raw train images copied into mixed train.")
    args = ap.parse_args()

    raw_yaml = Path(args.raw_data).resolve()
    base = yaml.safe_load(raw_yaml.read_text(encoding="utf-8"))
    names = _load_names(raw_yaml)

    raw_train_images = _resolve_from_yaml(raw_yaml, base, "train")
    raw_val_images = _resolve_from_yaml(raw_yaml, base, "val")
    raw_root = _resolve_root(raw_yaml, base, raw_train_images)
    raw_train_labels = raw_root / "labels" / "train"
    raw_val_labels = raw_root / "labels" / "val"
    if not raw_train_labels.exists():
        raise FileNotFoundError(f"Cannot find raw train labels: {raw_train_labels}")
    if not raw_val_labels.exists():
        raise FileNotFoundError(f"Cannot find raw val labels: {raw_val_labels}")

    enh_root = Path(args.enh_root).resolve()
    enh_train_images = enh_root / "images" / "train"
    enh_train_labels = enh_root / "labels" / "train"
    enh_val_images = enh_root / "images" / "val"
    enh_val_labels = enh_root / "labels" / "val"
    if not enh_train_images.exists() or not enh_train_labels.exists():
        raise FileNotFoundError("enh_root must contain images/train and labels/train")
    if not enh_val_images.exists() or not enh_val_labels.exists():
        raise FileNotFoundError("enh_root must contain images/val and labels/val")

    out_root = Path(args.out_root).resolve()
    train_images = out_root / "images" / "train"
    train_labels = out_root / "labels" / "train"
    val_images = out_root / "images" / "val"
    val_labels = out_root / "labels" / "val"
    _ensure_empty_dir(out_root)
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    raw_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    raw_train_list = []
    for p in raw_train_images.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in raw_exts:
            continue
        raw_train_list.append(p)
    raw_train_list = sorted(raw_train_list)
    if args.raw_train_limit and args.raw_train_limit > 0:
        raw_train_list = raw_train_list[: int(args.raw_train_limit)]

    for p in raw_train_list:
        shutil.copy2(p, train_images / p.name)
        lp = raw_train_labels / f"{p.stem}.txt"
        if lp.exists():
            shutil.copy2(lp, train_labels / lp.name)

    for p in enh_train_images.glob("*.png"):
        out_name = f"{p.stem}_enh.png"
        shutil.copy2(p, train_images / out_name)
        lp = enh_train_labels / f"{p.stem}.txt"
        if lp.exists():
            shutil.copy2(lp, train_labels / f"{p.stem}_enh.txt")

    for p in enh_val_images.glob("*.png"):
        shutil.copy2(p, val_images / p.name)
    for p in enh_val_labels.glob("*.txt"):
        shutil.copy2(p, val_labels / p.name)

    out_yaml = out_root / "data.yaml"
    _write_yaml(out_yaml, root=out_root, train_images=train_images, val_images=val_images, names=names)

    print("Wrote mixed dataset:", out_root)
    print("Train images:", len(list(train_images.glob('*'))))
    print("Val images:", len(list(val_images.glob('*.png'))))
    print("YAML:", out_yaml)


if __name__ == "__main__":
    main()

