import argparse
print("IMPORTING...")
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image


URPC_CLASSES = ["holothurian", "echinus", "scallop", "starfish"]


def _ensure_empty_dir(p: Path):
    print("Ensuring empty dir:", p)
    if p.exists():
        print("Removing tree:", p)
        shutil.rmtree(p)
    print("Making dir:", p)
    p.mkdir(parents=True, exist_ok=True)


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
        p = images_dir / f"{stem}{ext.upper()}"
        if p.exists():
            return p
    return None


def _parse_voc_xml(xml_path: Path, class_to_id: dict[str, int], img_path: Path | None = None):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        w = float(size.findtext("width", "0"))
        h = float(size.findtext("height", "0"))
    else:
        w, h = 0.0, 0.0

    if w <= 0 or h <= 0:
        if img_path is None or not img_path.exists():
            raise ValueError(f"Missing <size> and cannot infer image size for {xml_path}")
        with Image.open(img_path) as im:
            w, h = float(im.size[0]), float(im.size[1])

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name", "") or "").strip()
        if name not in class_to_id:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        xmin = max(0.0, min(xmin, w - 1.0))
        xmax = max(0.0, min(xmax, w - 1.0))
        ymin = max(0.0, min(ymin, h - 1.0))
        ymax = max(0.0, min(ymax, h - 1.0))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_c = ((xmin + xmax) / 2.0) / w
        y_c = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        cls_id = class_to_id[name]
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

    return lines


def main():
    print("STARTING SCRIPT")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Extracted URPC optical training dataset root folder.")
    ap.add_argument("--out", required=True, help="Output folder for YOLO dataset (will be overwritten).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--classes", default=",".join(URPC_CLASSES), help="Comma-separated class names in order.")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    print("Parsed args")
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    class_to_id = {c: i for i, c in enumerate(classes)}

    candidates = []

    images_dir = None
    ann_dir = None
    print("Globbing...")
    for p in [src, *src.glob("**/*")]:
        if p.is_dir() and p.name.lower() in {"jpegimages", "images", "image"}:
            images_dir = p
        if p.is_dir() and p.name.lower() in {"annotations", "annotation", "box"}:
            ann_dir = p
        if images_dir and ann_dir:
            break
    print("Glob done", images_dir, ann_dir)
    if images_dir is None or ann_dir is None:
        raise FileNotFoundError("Cannot locate VOC-style 'JPEGImages/images' and 'Annotations' folders under --src")

    xml_files = sorted(ann_dir.glob("*.xml"))
    print("Found", len(xml_files), "xml files")
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {ann_dir}")

    for i, x in enumerate(xml_files):
        stem = x.stem
        img = _find_image(images_dir, stem)
        if img is None:
            continue
        candidates.append((img, x))
        if i % 500 == 0:
            print("Matched", i, "candidates")
    print("Matched total", len(candidates), "candidates")

    if not candidates:
        raise RuntimeError("No (image, xml) pairs found.")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    n_val = int(round(len(candidates) * float(args.val_ratio)))
    val_set = candidates[:n_val]
    train_set = candidates[n_val:]

    images_train = out / "images" / "train"
    images_val = out / "images" / "val"
    labels_train = out / "labels" / "train"
    labels_val = out / "labels" / "val"
    _ensure_empty_dir(out)
    images_train.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    def _convert(split, split_images, split_labels):
        for i, (img_path, xml_path) in enumerate(split):
            lines = _parse_voc_xml(xml_path, class_to_id, img_path=img_path)
            shutil.copy2(img_path, split_images / img_path.name)
            (split_labels / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if i % 500 == 0:
                print("Converted", i, "files in split")
    print("Converting train set...")
    _convert(train_set, images_train, labels_train)
    print("Converting val set...")
    _convert(val_set, images_val, labels_val)

    data = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Prepared YOLO dataset at: {out}")
    print(f"Train images: {len(list(images_train.glob('*')))}; Val images: {len(list(images_val.glob('*')))}")
    print(f"data.yaml: {out / 'data.yaml'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("error.log", "w") as f:
            traceback.print_exc(file=f)
        print("EXCEPTION:", e)
