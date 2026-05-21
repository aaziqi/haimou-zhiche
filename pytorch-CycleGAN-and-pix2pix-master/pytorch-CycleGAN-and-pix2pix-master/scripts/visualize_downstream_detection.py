import argparse
import base64
import io
import os
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls_id: int
    conf: float | None = None


def _load_names(data_yaml: Path) -> list[str]:
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = d.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        out = []
        for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            out.append(str(names[k]))
        return out
    return []


def _read_yolo_labels(label_file: Path, w: int, h: int) -> list[Box]:
    if not label_file.exists():
        return []
    boxes: list[Box] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x = float(parts[1]) * w
        y = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h
        x1 = x - bw / 2
        y1 = y - bh / 2
        x2 = x + bw / 2
        y2 = y + bh / 2
        boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, cls_id=cls_id))
    return boxes


def _draw_boxes(img: Image.Image, boxes: list[Box], *, names: list[str], color: tuple[int, int, int], width: int, show_conf: bool):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for b in boxes:
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=width)
        label = names[b.cls_id] if 0 <= b.cls_id < len(names) else str(b.cls_id)
        if show_conf and b.conf is not None:
            label = f"{label} {b.conf:.2f}"
        if font is not None:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            tx, ty = b.x1, max(0, b.y1 - th - 2)
            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=(0, 0, 0))
            draw.text((tx + 2, ty + 1), label, fill=(255, 255, 255), font=font)


def _predict_boxes(model, img: Image.Image, conf: float, imgsz: int) -> list[Box]:
    r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
    if not r:
        return []
    res = r[0]
    if getattr(res, "boxes", None) is None:
        return []
    out: list[Box] = []
    xyxy = res.boxes.xyxy
    cls = res.boxes.cls
    confs = res.boxes.conf
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(x) for x in xyxy[i].tolist()]
        out.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, cls_id=int(cls[i].item()), conf=float(confs[i].item())))
    return out


def _make_triptych(a: Image.Image, b: Image.Image, c: Image.Image, titles: tuple[str, str, str]) -> Image.Image:
    w, h = a.size
    out = Image.new("RGB", (w * 3, h + 28), (255, 255, 255))
    out.paste(a, (0, 28))
    out.paste(b, (w, 28))
    out.paste(c, (w * 2, 28))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, t in enumerate(titles):
        x = i * w + 6
        y = 6
        if font is not None:
            draw.text((x, y), t, fill=(0, 0, 0), font=font)
        else:
            draw.text((x, y), t, fill=(0, 0, 0))
    return out


def _make_pair(a: Image.Image, b: Image.Image, titles: tuple[str, str]) -> Image.Image:
    w, h = a.size
    out = Image.new("RGB", (w * 2, h + 28), (255, 255, 255))
    out.paste(a, (0, 28))
    out.paste(b, (w, 28))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, t in enumerate(titles):
        x = i * w + 6
        y = 6
        if font is not None:
            draw.text((x, y), t, fill=(0, 0, 0), font=font)
        else:
            draw.text((x, y), t, fill=(0, 0, 0))
    return out


def _img_to_data_uri(p: Path, max_w: int = 900) -> str:
    im = Image.open(p).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + b64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO data.yaml (for class names + label root).")
    ap.add_argument("--weights", required=True, help="Detector weights (best.pt).")
    ap.add_argument("--raw_images", required=True, help="Raw val images folder (images/val).")
    ap.add_argument("--raw_labels", required=True, help="Raw val labels folder (labels/val).")
    ap.add_argument("--cyc_images", default="", help="CycleGAN enhanced images folder (png). Optional.")
    ap.add_argument("--mpc_images", required=True, help="Enhanced images folder (png).")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ultra_cfg = out / "ultralytics_config"
    ultra_cfg.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = os.environ.get("YOLO_CONFIG_DIR", str(ultra_cfg))

    from ultralytics import YOLO

    data_yaml = Path(args.data).resolve()
    names = _load_names(data_yaml)

    raw_images = Path(args.raw_images).resolve()
    raw_labels = Path(args.raw_labels).resolve()
    cyc_images = Path(args.cyc_images).resolve() if args.cyc_images else None
    mpc_images = Path(args.mpc_images).resolve()
    raw_candidates = sorted([p for p in raw_images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if cyc_images is None:
        stems = [p.stem for p in raw_candidates if (mpc_images / f"{p.stem}.png").exists()]
    else:
        stems = [p.stem for p in raw_candidates if (cyc_images / f"{p.stem}.png").exists() and (mpc_images / f"{p.stem}.png").exists()]
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    stems = stems[: max(1, int(args.n))]

    model = YOLO(str(Path(args.weights).resolve()))

    rendered: list[Path] = []
    for stem in stems:
        raw_p = raw_images / f"{stem}.jpg"
        if not raw_p.exists():
            raw_p = raw_images / f"{stem}.png"
        mpc_p = mpc_images / f"{stem}.png"
        if not raw_p.exists() or not mpc_p.exists():
            continue

        raw_im = Image.open(raw_p).convert("RGB")
        mpc_im = Image.open(mpc_p).convert("RGB")

        w, h = raw_im.size
        if mpc_im.size != (w, h):
            mpc_im = mpc_im.resize((w, h), resample=Image.BICUBIC)
        gt = _read_yolo_labels(raw_labels / f"{stem}.txt", w=w, h=h)

        raw_pred = _predict_boxes(model, raw_im, conf=float(args.conf), imgsz=int(args.imgsz))
        mpc_pred = _predict_boxes(model, mpc_im, conf=float(args.conf), imgsz=int(args.imgsz))

        raw_vis = raw_im.copy()
        mpc_vis = mpc_im.copy()
        _draw_boxes(raw_vis, gt, names=names, color=(0, 255, 0), width=2, show_conf=False)
        _draw_boxes(mpc_vis, gt, names=names, color=(0, 255, 0), width=2, show_conf=False)

        _draw_boxes(raw_vis, raw_pred, names=names, color=(255, 0, 0), width=2, show_conf=True)
        _draw_boxes(mpc_vis, mpc_pred, names=names, color=(255, 0, 0), width=2, show_conf=True)

        if cyc_images is None:
            pair = _make_pair(raw_vis, mpc_vis, ("RAW (GT+Pred)", "Enhanced (GT+Pred)"))
            out_p = out / f"{stem}_pair.jpg"
            pair.save(out_p, quality=90)
        else:
            cyc_p = cyc_images / f"{stem}.png"
            if not cyc_p.exists():
                continue
            cyc_im = Image.open(cyc_p).convert("RGB")
            if cyc_im.size != (w, h):
                cyc_im = cyc_im.resize((w, h), resample=Image.BICUBIC)
            cyc_pred = _predict_boxes(model, cyc_im, conf=float(args.conf), imgsz=int(args.imgsz))
            cyc_vis = cyc_im.copy()
            _draw_boxes(cyc_vis, gt, names=names, color=(0, 255, 0), width=2, show_conf=False)
            _draw_boxes(cyc_vis, cyc_pred, names=names, color=(255, 0, 0), width=2, show_conf=True)
            trip = _make_triptych(raw_vis, cyc_vis, mpc_vis, ("RAW (GT+Pred)", "CycleGAN (GT+Pred)", "MP-CycleGAN (GT+Pred)"))
            out_p = out / f"{stem}_triptych.jpg"
            trip.save(out_p, quality=90)
        rendered.append(out_p)

    cards = []
    for p in rendered:
        uri = _img_to_data_uri(p)
        cards.append(f"<div style='margin:12px 0'><div style='font-family:monospace'>{p.name}</div><img src='{uri}' style='max-width:100%;border:1px solid #ddd'/></div>")

    html = (
        "<html><head><meta charset='utf-8'><title>Downstream Detection Visualization</title></head><body>"
        "<h2>Downstream Detection Visualization</h2>"
        "<div>Green: GT boxes. Red: detector predictions.</div>"
        + "\n".join(cards)
        + "</body></html>"
    )
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {len(rendered)} visualizations to: {out}")
    print(f"Index: {out / 'index.html'}")


if __name__ == "__main__":
    main()

