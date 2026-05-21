import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen


def post_json(url: str, body: dict, timeout: int):
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:8877")
    ap.add_argument("--image", required=True)
    ap.add_argument("--model_key", default="improved_mpcgan")
    ap.add_argument("--detector_key", default="default_detector")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    img_path = Path(args.image).resolve()
    data_url = "data:image/png;base64," + base64.b64encode(img_path.read_bytes()).decode("ascii")
    enh_payload = post_json(
        args.base_url + "/api/enhance",
        {"filename": img_path.name, "image": data_url, "model_key": args.model_key},
        timeout=args.timeout,
    )
    if not enh_payload.get("ok"):
        raise SystemExit(f"enhance failed: {enh_payload}")
    raw_url = enh_payload["result"]["input_url"]
    enh_url = enh_payload["result"]["output_url"]

    det_payload = post_json(
        args.base_url + "/api/detect_compare",
        {"raw_image_url": raw_url, "enhanced_image_url": enh_url, "detector_key": args.detector_key, "conf": args.conf, "imgsz": args.imgsz},
        timeout=args.timeout,
    )
    if not det_payload.get("ok"):
        raise SystemExit(f"detect_compare failed: {det_payload}")

    print("enhance_ok", enh_payload.get("ok"))
    print("raw_url", raw_url)
    print("enh_url", enh_url)
    print("detect_compare_ok", det_payload.get("ok"))
    print("raw_dets", len(det_payload.get("raw_detections") or []))
    print("enh_dets", len(det_payload.get("enhanced_detections") or []))
    print("compare_vis_url", det_payload.get("compare_vis_url"))
    print("raw_json_url", det_payload.get("raw_json_url"))
    print("enh_json_url", det_payload.get("enhanced_json_url"))
    print("report_url", det_payload.get("report_url"))


if __name__ == "__main__":
    main()

