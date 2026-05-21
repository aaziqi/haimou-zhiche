import argparse
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def _rd(metrics) -> dict:
    d = getattr(metrics, "results_dict", None) or {}
    out = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            out[str(k)] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--raw_data", required=True)
    ap.add_argument("--enh_data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["KMP_DUPLICATE_LIB_OK"] = os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    os.environ["YOLO_CONFIG_DIR"] = os.environ.get("YOLO_CONFIG_DIR", str(out.parent / "ultralytics_config"))

    from ultralytics import YOLO

    log_path = out.parent / "ultralytics_val.log"
    with log_path.open("w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
        model = YOLO(str(Path(args.weights).resolve()))
        m_raw = model.val(data=str(Path(args.raw_data).resolve()), imgsz=int(args.imgsz), device=args.device, verbose=False, plots=False, workers=0)
        m_enh = model.val(data=str(Path(args.enh_data).resolve()), imgsz=int(args.imgsz), device=args.device, verbose=False, plots=False, workers=0)

    payload = {
        "weights": str(Path(args.weights).resolve()),
        "raw_data": str(Path(args.raw_data).resolve()),
        "enh_data": str(Path(args.enh_data).resolve()),
        "imgsz": int(args.imgsz),
        "device": args.device,
        "metrics_raw": _rd(m_raw),
        "metrics_enhanced": _rd(m_enh),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()

