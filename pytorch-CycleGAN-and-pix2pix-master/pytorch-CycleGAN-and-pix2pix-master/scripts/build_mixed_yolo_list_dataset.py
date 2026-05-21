import argparse
import random
from pathlib import Path

import yaml


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


def _load_names(base: dict) -> list[str]:
    names = base.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
    return [str(x) for x in names]


def _collect_images(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_data", required=True, help="Original YOLO data.yaml (raw train/val + labels).")
    ap.add_argument("--enh_root", required=True, help="Enhanced dataset root (must contain images/train and labels/train).")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--raw_train_limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    raw_yaml = Path(args.raw_data).resolve()
    base = yaml.safe_load(raw_yaml.read_text(encoding="utf-8"))
    names = _load_names(base)

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
    if not enh_train_images.exists() or not enh_train_labels.exists():
        raise FileNotFoundError("enh_root must contain images/train and labels/train")

    raw_imgs = _collect_images(raw_train_images)
    if args.raw_train_limit and args.raw_train_limit > 0:
        rng = random.Random(int(args.seed))
        rng.shuffle(raw_imgs)
        raw_imgs = sorted(raw_imgs[: int(args.raw_train_limit)])

    enh_imgs = sorted([p for p in enh_train_images.glob("*.png") if p.is_file()])

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_txt = out_root / "train.txt"
    val_txt = out_root / "val.txt"
    data_yaml = out_root / "data.yaml"

    train_lines = [str(p) for p in raw_imgs] + [str(p) for p in enh_imgs]
    train_txt.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    val_lines = [str(p) for p in _collect_images(raw_val_images)]
    val_txt.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    d = {
        "path": str(out_root),
        "train": str(train_txt),
        "val": str(val_txt),
        "nc": len(names),
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")

    print("Wrote:", data_yaml)
    print("Train images:", len(train_lines))
    print("Val images:", len(val_lines))


if __name__ == "__main__":
    main()

