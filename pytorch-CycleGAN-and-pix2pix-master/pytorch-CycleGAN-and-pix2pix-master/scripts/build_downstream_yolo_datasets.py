import argparse
import shutil
from pathlib import Path

import yaml


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _copy_labels(src_labels_dir: Path, dst_labels_dir: Path):
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    for p in src_labels_dir.glob("*.txt"):
        shutil.copy2(p, dst_labels_dir / p.name)


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
    ap.add_argument("--raw_data", required=True, help="Original YOLO data.yaml (provides names + label layout).")
    ap.add_argument("--mpcgan_images_dir", required=True, help="CycleGAN test output images dir containing *_fake.png.")
    ap.add_argument("--mpcgan_train_subset", required=True, help="Enhanced train subset root with images/train and labels/train.")
    ap.add_argument("--out_root", required=True, help="Output dataset root.")
    args = ap.parse_args()

    raw_data = Path(args.raw_data).resolve()
    base = yaml.safe_load(raw_data.read_text(encoding="utf-8"))
    names = base.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
    names = [str(x) for x in names]

    val_images = Path(base["val"])
    if not val_images.is_absolute():
        val_images = (raw_data.parent / val_images).resolve()

    train_images = Path(base["train"])
    if not train_images.is_absolute():
        train_images = (raw_data.parent / train_images).resolve()

    if "path" in base:
        raw_root = Path(base["path"])
        if not raw_root.is_absolute():
            raw_root = (raw_data.parent / raw_root).resolve()
    else:
        raw_root = train_images.parent.parent

    raw_val_labels = raw_root / "labels" / "val"
    if not raw_val_labels.exists():
        raise FileNotFoundError(f"Cannot find raw val labels: {raw_val_labels}")

    raw_train_labels = raw_root / "labels" / "train"
    if not raw_train_labels.exists():
        raise FileNotFoundError(f"Cannot find raw train labels: {raw_train_labels}")

    mpcgan_images_dir = Path(args.mpcgan_images_dir).resolve()
    if not mpcgan_images_dir.exists():
        raise FileNotFoundError(f"mpcgan_images_dir not found: {mpcgan_images_dir}")

    subset_root = Path(args.mpcgan_train_subset).resolve()
    subset_train_images = subset_root / "images" / "train"
    subset_train_labels = subset_root / "labels" / "train"
    if not subset_train_images.exists() or not subset_train_labels.exists():
        raise FileNotFoundError("mpcgan_train_subset must contain images/train and labels/train")

    out_root = Path(args.out_root).resolve()
    images_train = out_root / "images" / "train"
    labels_train = out_root / "labels" / "train"
    images_val = out_root / "images" / "val"
    labels_val = out_root / "labels" / "val"
    _ensure_dir(images_train)
    _ensure_dir(labels_train)
    _ensure_dir(images_val)
    _ensure_dir(labels_val)

    for p in subset_train_images.glob("*.png"):
        dst = images_train / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        lp = subset_train_labels / f"{p.stem}.txt"
        if lp.exists():
            dst_l = labels_train / lp.name
            if not dst_l.exists():
                shutil.copy2(lp, dst_l)

    for p in mpcgan_images_dir.glob("*_fake.png"):
        stem = p.name[: -len("_fake.png")]
        dst = images_val / f"{stem}.png"
        if not dst.exists():
            shutil.copy2(p, dst)
        lp = raw_val_labels / f"{stem}.txt"
        if lp.exists():
            dst_l = labels_val / lp.name
            if not dst_l.exists():
                shutil.copy2(lp, dst_l)

    out_yaml = out_root / "data.yaml"
    _write_yaml(out_yaml, root=out_root, train_images=images_train, val_images=images_val, names=names)

    print("Wrote dataset:", out_root)
    print("Train images:", len(list(images_train.glob("*.png"))))
    print("Val images:", len(list(images_val.glob("*.png"))))
    print("YAML:", out_yaml)


if __name__ == "__main__":
    main()

