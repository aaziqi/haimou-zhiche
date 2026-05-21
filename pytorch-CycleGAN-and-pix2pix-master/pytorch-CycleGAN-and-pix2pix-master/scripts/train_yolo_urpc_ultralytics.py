import argparse
import traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="yolov8n_urpc")
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    Path(args.project).mkdir(parents=True, exist_ok=True)
    m = YOLO(args.model)
    m.train(
        data=str(Path(args.data)),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        device=args.device,
        project=str(Path(args.project)),
        name=args.name,
        exist_ok=True,
        amp=False,
        plots=False,
        verbose=True,
        workers=0,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        Path("train_yolo_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise

