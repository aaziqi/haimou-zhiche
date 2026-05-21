import argparse
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["KMP_DUPLICATE_LIB_OK"] = os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    os.environ["YOLO_CONFIG_DIR"] = os.environ.get("YOLO_CONFIG_DIR", str(out.parent / "ultralytics_config"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    log_path = out.parent / (out.stem + ".log")
    out_dict = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "imgsz": int(args.imgsz),
        "device": args.device,
    }
    try:
        with log_path.open("w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
            model = YOLO(str(Path(args.weights).resolve()))
            metrics = model.val(
                data=str(Path(args.data).resolve()),
                imgsz=int(args.imgsz),
                device=args.device,
                verbose=False,
                plots=False,
                workers=0,
            )
            d = getattr(metrics, "results_dict", None) or {}
        out_dict["metrics"] = {str(k): float(v) for k, v in d.items()}
    except BaseException as e:
        out_dict["error"] = str(e)
        out_dict["traceback"] = traceback.format_exc()

    out.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()

