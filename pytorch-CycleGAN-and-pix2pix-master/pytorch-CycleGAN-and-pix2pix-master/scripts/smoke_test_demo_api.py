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
    output_url = enh_payload["result"]["output_url"]

    det_payload = post_json(
        args.base_url + "/api/detect",
        {"image_url": output_url, "detector_key": args.detector_key, "conf": args.conf, "imgsz": args.imgsz},
        timeout=args.timeout,
    )
    if not det_payload.get("ok"):
        raise SystemExit(f"detect failed: {det_payload}")

    print("enhance_ok", enh_payload.get("ok"))
    print("output_url", output_url)
    print("detect_ok", det_payload.get("ok"))
    print("detections", len(det_payload.get("detections") or []))
    print("vis_url", det_payload.get("vis_url"))
    print("json_url", det_payload.get("json_url"))


if __name__ == "__main__":
    main()
